"""Inductor kernel  ->  tnpu KernelSpec.

Two jobs, both of which exist because Inductor's Triton output is written for a
GPU launcher and tnpu's is written for a static, ahead-of-time pipeline:

1. `collect_meta` -- pull everything tnpu needs out of the Inductor kernel while
   we still have `V.graph`: argument names/roles/dtypes/sizes, the constexprs,
   and the numels the grid is computed from. This runs at codegen time; by the
   time the compile callable fires, `V.graph` is gone.

2. `write_spec_file` -- turn the Triton source + that metadata into a kernel file
   tnpu can load (`tnpu.spec.load_spec`).

WHY THE SOURCE HAS TO BE REWRITTEN
----------------------------------
Inductor emits, above the kernel:

    from torch._inductor.runtime import triton_heuristics
    @triton_heuristics.pointwise(size_hints={'x': 1024}, ..., inductor_meta=...)
    @triton.jit
    def triton_npu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):

Neither line can survive into tnpu:

  * the tnpu triton venv has NO torch (deliberately -- tnpu/spec.py), so the
    `torch._inductor.runtime` import fails on sight;
  * `triton_heuristics.pointwise` is the AUTOTUNER. It picks XBLOCK at runtime
    and derives `grid = cdiv(xnumel, XBLOCK)` from it. tnpu needs both to be
    constants: the block size becomes a `tl.constexpr` in the ttir signature and
    the grid is executed as a sequential loop by the generated C wrapper.

So the decorator is stripped and XBLOCK is pinned as a constexpr instead. That is
not a workaround -- fixing the config at codegen time is what makes the kernel
statically describable, which is the whole premise of this route.
"""

import math
import os
import re

from torch._inductor.virtualized import V

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

#: Triton signature token -> (torch dtype name, bytes). Only the dtypes
#: tnpu/wrapper.py can round-trip through .raw files.
_DTYPE = {
    "*fp64": "float64", "*fp32": "float32", "*fp16": "float16",
    "*bf16": "bfloat16",
    "*i64": "int64", "*i32": "int32", "*i16": "int16", "*i8": "int8",
    "*u64": "uint64", "*u32": "uint32", "*u16": "uint16", "*u8": "uint8",
    "*i1": "bool",
    "fp32": "float32", "i32": "int32", "i64": "int64",
}


#: Triton scalar token -> C type, for the wrapper's kernel declaration.
_C_TYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}


class SpecIncomplete(RuntimeError):
    """Metadata tnpu requires that this kernel did not provide.

    Raised with the missing field named, rather than writing a spec that fails
    deeper in the pipeline where the cause is unrecoverable.
    """


# ---------------------------------------------------------------------------
# 1. codegen-time metadata capture
# ---------------------------------------------------------------------------
def _buffer_numel(name):
    """Element count of an Inductor buffer, or None if it cannot be resolved."""
    try:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None
        size = buf.get_layout().size
        n = 1
        for s in size:
            n *= int(V.graph.sizevars.size_hint(s))
        return n
    except Exception:  # noqa: BLE001 - best effort; caller reports it as missing
        return None


def _roles(kernel):
    """arg name -> 'in' | 'out' | 'inout', from the kernel's buffer tables."""
    out = {}
    for buf, arg in getattr(kernel.args, "input_buffers", {}).items():
        out[arg] = ("in", buf)
    for buf, arg in getattr(kernel.args, "output_buffers", {}).items():
        out[arg] = ("out", buf)
    for buf, arg in getattr(kernel.args, "inplace_buffers", {}).items():
        name = getattr(arg, "inner_name", arg)
        out[name] = ("inout", buf)
    return out


def collect_meta(kernel, kernel_name):
    """Everything the compile step needs, as plain repr-able data.

    Must run while `V.graph` is live (i.e. inside define_kernel).
    """
    triton_meta = dict(getattr(kernel, "triton_meta", None) or {})
    signature = dict(triton_meta.get("signature") or {})
    constants = dict(triton_meta.get("constants") or {})

    roles = _roles(kernel)
    arg_defs, _call_args, _precompile, _arg_types = kernel.args.python_argdefs()

    args = []
    for a in arg_defs:
        name = getattr(a, "name", str(a))
        role, buf = roles.get(name, (None, None))
        if role is None:
            continue                      # a numel / constexpr, not a tensor
        args.append({
            "name": name,
            "role": role,
            "buffer": buf,
            "dtype": _DTYPE.get(signature.get(name, ""), None),
            "numel": _buffer_numel(buf) if buf else None,
        })

    # The numels Inductor appends to the call. They live in `kernel.numels`,
    # keyed by iteration-space PREFIX ('x', 'y', 'r0', ...), not as xnumel/rnumel
    # attributes (SIMDKernel.__init__ builds them from the tiling). These are
    # what the grid is computed from.
    numels = {}
    for prefix, val in (getattr(kernel, "numels", None) or {}).items():
        try:
            numels[f"{prefix}numel"] = int(V.graph.sizevars.size_hint(val))
        except Exception:  # noqa: BLE001 - dynamic shape; reported by _grid
            numels[f"{prefix}numel"] = None

    return {
        "kernel_name": kernel_name,
        "template_grid": _template_grid(kernel),
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": fixed_config_for(kernel, numels, args),
    }


#: Parallel iteration prefixes, OUTERMOST first. Inductor's `x` is the
#: contiguous axis, so it is innermost; `r*` prefixes are reductions, looped
#: inside the kernel rather than spread over the grid (prefix_is_reduction).
_PARALLEL_PREFIXES = ("z", "y", "x")


def _block_name(prefix):
    return f"{prefix.upper()}BLOCK"


def parallel_axes(numels):
    """Grid axes this kernel uses, outermost first."""
    return [p for p in _PARALLEL_PREFIXES if f"{p}numel" in numels]


def reduction_axes(numels):
    """Reduction prefixes, read off the keys rather than assumed.

    THE PREFIX CARRIES A TRAILING UNDERSCORE: Inductor names the parameter
    `r0_numel` and the block `R0_BLOCK`, so the prefix is "r0_" and not "r0".
    Looking up "r0numel" silently misses and leaves the block unset, which
    surfaces much later as "block size R0_BLOCK is unset" on a kernel whose
    extent was known all along. Derived here so the spelling is read once.
    """
    return sorted(k[:-len("numel")] for k in numels
                  if k.endswith("numel") and k.startswith("r"))


#: How many R0_BLOCK-sized buffers a reduction body keeps live at once.
#:
#: R0_BLOCK IS PER LANE. The tile is [XBLOCK, R0_BLOCK], XBLOCK is the lane axis,
#: so every intermediate costs R0_BLOCK elements IN EVERY LANE and the block size
#: multiplies by however many are live. DeepSeek's RMSNorm lowers to ten spad
#: globals, eight of them full R0_BLOCK tiles -- and one of those is banked on a
#: unit axis, which reserves its full size in every lane anyway, so it counts too.
#:
#: 12, not 8. Eight was the measured count and it landed 64 bytes over budget,
#: because the count is a property of the kernel and not a constant. The cost of
#: guessing high is one more trip round a loop that already exists; the cost of
#: guessing low is a link-time scratchpad error that never mentions block sizes.
_REDUCTION_LIVE_TILES = 12


#: torch dtype name -> bits. Only the dtypes _DTYPE can name.
_DTYPE_BITS = {"float64": 64, "float32": 32, "float16": 16, "bfloat16": 16,
               "int64": 64, "int32": 32, "int16": 16, "int8": 8,
               "uint64": 64, "uint32": 32, "uint16": 16, "uint8": 8,
               "bool": 8}


def _element_bits(args):
    """Widest tensor element in the kernel, in bits.

    Widest, not narrowest: the block has to be one register's worth for EVERY
    operand, and sizing off a narrow one would ask the wide one for more lanes
    than a register holds. A kernel with no typed tensor argument falls back to
    32, which is what every current baseline is.
    """
    bits = [_DTYPE_BITS[a["dtype"]] for a in (args or [])
            if a.get("dtype") in _DTYPE_BITS]
    return max(bits) if bits else 32


def fixed_config_for(kernel, numels, args):
    """Block sizes pinned at codegen time.

    `numels` is the <prefix>numel-keyed dict collect_meta builds, NOT
    kernel.numels, which is keyed by bare prefix. Deriving it here a second time
    is what went wrong before: this passed kernel.numels straight to
    parallel_axes, which looks for "ynumel" and found "y", so axes came back
    EMPTY for every kernel. Single-axis ones survived on the XBLOCK setdefault
    below and multi-axis ones lost YBLOCK entirely, surfacing much later as
    "YBLOCK=None" out of grid_of. The two must read the same dict.

    tnpu compiles ONE binary per kernel and the C wrapper walks the grid as a
    sequential loop, so there is no autotuner to choose the blocks later and no
    runtime `grid=` callable. Fixing them here is what makes the launch shape
    static.

    THESE ARE SIZES, NOT A LAYOUT. Nothing here chooses which axis lands on the
    lanes, because that is not decided until tnpu's select_lane_axis, several
    passes into the other repo: it gathers what every op demands (a matmul wants
    its last axis on the lanes, a reduction wants the axis it folds off them,
    elementwise operands have to agree), resolves conflicts by flipping the
    matmul, and only defaults to dim 0 when nothing asked. Measured over the
    tnpu dumps: 26 of 36 stamped kernels are axis 0, gemm_fp16_kernel is axis 1
    throughout, and gemm/bmm carry BOTH on different operands of one op.

    THIS USED TO SAY DIM 0 WAS THE LANE AXIS AND THAT A LANE MUST HOLD EXACTLY
    ONE ELEMENT. Both halves were wrong, and the second is why the first looked
    right: pinning the outermost block to the lane count made dim 0 the
    degenerate answer often enough that the claim was never tested. It is tested
    now -- kernels/coverage/tile/tile_deeper_than_one_per_lane.py runs a
    256-wide tile on 128 lanes exactly, and tile_gemm_lane_axis_deeper.py does
    it on the axis a matmul demanded, at 9.54e-06 against a 1.53e-05 control.
    Nine memref<128x256xf32, 1> spad buffers in that kernel's 04-adapted.mlir,
    so the deeper tile is really built and really addressed.

    So the lane count below is a DEFAULT WIDTH -- a tile at least as wide as the
    machine -- and not a requirement the backend would break without. Choosing
    real tile sizes is still the block-size policy gap in README; what has gone
    away is the reason to believe there was only one legal answer.
    """
    from . import tnpu_bridge
    machine = tnpu_bridge.machine()
    lanes = machine["lanes"]
    per_vector = max(1, machine["vlen_bits"] // _element_bits(args))

    def _covers(want, n):
        """`want`, clamped to the smallest legal block that covers `n`.

        A BLOCK BIGGER THAN ITS NUMEL IS NOT FREE, and it used to be handed out
        unconditionally. `t.sum(dim=0)` on a [128, 64] tensor gives xnumel 64
        against XBLOCK 128, so the tile is twice the iteration space and the
        surplus is carried by a mask -- whose rank-1 index rows are then
        stranded ONE_LANE while the data is banked across the lanes, and stage 4
        stops:

            NotVectorisable: one operand of this elementwise op is in a single
            bank (ONE_LANE) while another is banked across the lanes on
            iteration dim 0 of extent 128

        `sum(dim=1)` on the same tensor has xnumel 128 against the same XBLOCK,
        needs no mask at all, and comes back at 1.907e-06. So the axis was never
        the difference; the block overshooting its numel was.

        Rounded UP to a power of two because `tl.arange` needs one, so a numel
        of 100 still gets 128 and still needs its mask -- that one is real.

        AND THE BACKEND'S SIDE IS NARROWER THAN ITS MESSAGE. A hand-written tnpu
        kernel that reduces over the banked axis with a block deliberately wider
        than the extent -- [128, 64] data under a [128, 128] block, so the mask
        genuinely fires -- COMPILES AND IS EXACT. Written with the block equal
        to the extent it also passes, because a trivially true mask folds away
        before stage 4 sees it. So the refusal above is not "a mask over a
        banked axis" in general, and no coverage kernel was added for it: one
        that passes either way pins nothing. What is measured is this: the
        oversized block was the whole observable, and clamping it is the fix.
        """
        if not n or n <= 0:
            return want
        cover = 1
        while cover < n:
            cover *= 2
        return min(want, cover)

    axes = parallel_axes(numels)
    cfg = {}
    for i, p in enumerate(axes):
        if i == 0:
            # The widest block goes on the outermost axis. NOT because that axis
            # is the lane axis -- see the docstring; tnpu decides that later and
            # picks 1 for every gemm it was measured on -- but because the tile
            # has to be wide SOMEWHERE for the machine to be busy, and with the
            # lane axis unknown at this point one axis is as good a guess as
            # another. Outermost is the arbitrary half of this; the lane count
            # is the part that is about the machine.
            #
            # WHICH LETTER dim 0 IS DEPENDS ON RANK, so it is indexed rather
            # than named. Inductor gives the outermost axis the [:, None] slot,
            # so a 2D kernel puts y there and a 1D one has only x.
            cfg[_block_name(p)] = _covers(lanes, numels.get(f"{p}numel"))
        elif i == len(axes) - 1:
            # Innermost, so contiguous (tile_stride ends in 1): give each lane
            # one full vector register. 8 fp32 at vlen 256 is exactly that.
            # Pinned to 1 before, which is why a multi-axis work-item moved a
            # strided column -- correct but nothing worth measuring.
            cfg[_block_name(p)] = _covers(per_vector,
                                          numels.get(f"{p}numel"))
        else:
            cfg[_block_name(p)] = 1
    cfg.setdefault("XBLOCK", _covers(lanes, numels.get("xnumel")))
    # a kernel with no tiling info still has x
    if getattr(kernel, "inside_reduction", False):
        # A reduction block is NOT free to be the lane count. The scratchpad is
        # lane-banked and there is no lane-crossing primitive, so the reduced
        # axis has to live INSIDE one lane (triton-npu kernels/reduce.py) --
        # spreading it over the lanes is the one layout the hardware cannot
        # execute. That leaves exactly one choice: cover the whole extent, so
        # every work-item reduces a complete row and nothing has to be
        # accumulated across grid steps (each step is independent here -- the
        # wrapper walks the grid as a plain loop with no carried state).
        #
        # Rounded up to a power of two because triton requires it; the tail is
        # masked. Left unset, as before, when the extent is unknown or will not
        # fit a lane's scratchpad -- those fail loudly rather than silently
        # picking a layout that cannot run.
        # INDUCTOR ALREADY LOOPS. The generated body is
        # `for r0_offset in range(0, r0_numel, R0_BLOCK)` with the partial sums
        # carried in registers, so the block does NOT have to cover the extent --
        # it only has to be a tile the hardware can hold. Sizing it to the whole
        # reduction, which an earlier version of this did, makes a 1536-long
        # reduction ask for 8KB per lane per buffer, and a kernel with five live
        # buffers then blows the double-buffer budget at link time with an error
        # that names the scratchpad rather than this decision.
        #
        # So take a fixed tile and let the loop do the rest. The reduced axis
        # still lives INSIDE a lane, which is the constraint that matters
        # (triton-npu kernels/reduce.py) -- a chunk of it is as in-lane as all
        # of it. The budget is halved because the target double-buffers, and
        # divided again by the number of tiles a reduction body keeps live.
        # NOT extension_config's spad. The YAML says 128 KB per lane and tnpu
        # enforces 64 KB (tnpu/config.py SPAD_SIZE, and spike is launched with
        # --scratchpad-size=65536), so sizing against the YAML overshoots by 2x
        # and the kernel dies at LINK time. tnpu is the one that refuses, so its
        # number is the one that counts; TNPU_SPAD_SIZE overrides both.
        lane_bytes = machine["spad_size"]
        elem_bytes = max(1, _element_bits(args) // 8)
        budget = lane_bytes // 2 // _REDUCTION_LIVE_TILES
        for p in reduction_axes(numels) or ["r0_"]:
            n = numels.get(f"{p}numel")
            if not n:
                cfg[_block_name(p)] = None
                continue
            # Cover the extent and no more, then shrink to the budget. Rounding
            # 1536 up to 2048 buys nothing -- the tail is all mask -- and costs a
            # third of the tile, so the loop running twice over 1024 is strictly
            # better than once over 2048 plus 512 wasted lanes of nothing.
            block = 1 << (int(n) - 1).bit_length()
            while block * elem_bytes > budget and block > 1:
                block //= 2
            cfg[_block_name(p)] = block
    return cfg


# ---------------------------------------------------------------------------
# 2. Triton source -> tnpu kernel file
# ---------------------------------------------------------------------------
_HEURISTIC_RE = re.compile(r"^@triton_heuristics\.")
_DROP_IMPORT_RE = re.compile(
    r"^\s*(import torch|from torch\b|from __future__|import __main__)")
#: GPU-only runtime setup Inductor emits at module scope. Meaningless here (the
#: kernel is compiled ahead of time to a RISC-V ELF) and its import is dropped
#: above, so the call would be a NameError.
_DROP_CALL_RE = re.compile(r"^\s*triton_helpers\.set_driver_to_gpu\(\)")
#: Anything else from triton_helpers is a real dependency -- maximum/minimum/
#: promote_to_tensor and friends, which reductions and clamps use constantly.
_HELPER_USE_RE = re.compile(r"\btriton_helpers\.(\w+)")

# Vendored verbatim from torch._inductor.runtime.triton_helpers. These are pure
# triton -- @triton.jit over tl.* and nothing else -- so the torch-free tnpu venv
# can run them; that is the whole reason a copy is possible at all, and it is
# also the test for whether a helper belongs here. Anything reaching into torch
# does not, and stays a SpecIncomplete below.
#
# The NaN handling is the point and must not be "simplified" to tl.maximum:
# `mask |= a != a` makes a NaN operand win, which is what torch.maximum promises
# and what tl.maximum does not do.
_VENDORED_HELPERS = {"promote_to_tensor", "is_floating",
                     "minimum", "maximum", "min2", "max2", "any",
                     "welford_reduce", "welford_combine", "welford"}

_HELPERS_SRC = '''
# Self-sufficient on purpose: this block is prepended, so it runs BEFORE the
# kernel body's own imports and cannot borrow them. Re-importing is free.
import types as _types
import triton
import triton.language as tl

@triton.jit
def _tnpu_promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)

@triton.jit
def _tnpu_is_floating(x):
    return _tnpu_promote_to_tensor(x).dtype.is_floating()

@triton.jit
def _tnpu_minimum(a, b):
    mask = a < b
    if _tnpu_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

@triton.jit
def _tnpu_maximum(a, b):
    mask = a > b
    if _tnpu_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

@triton.jit
def _tnpu_min2(a, dim):
    return tl.reduce(a, dim, _tnpu_minimum)

@triton.jit
def _tnpu_max2(a, dim):
    return tl.reduce(a, dim, _tnpu_maximum)

@triton.jit
def _tnpu_any_combine(a, b):
    return a | b

@triton.jit
def _tnpu_any(a, dim):
    return tl.reduce(a, dim, _tnpu_any_combine)

@triton.jit
def _tnpu_welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight

@triton.jit
def _tnpu_welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )

@triton.jit
def _tnpu_welford(mean, m2, weight, dim):
    return tl.reduce((mean, m2, weight), dim, _tnpu_welford_combine)

triton_helpers = _types.ModuleType("triton_helpers")
triton_helpers.any = _tnpu_any
triton_helpers.welford_reduce = _tnpu_welford_reduce
triton_helpers.welford_combine = _tnpu_welford_combine
triton_helpers.welford = _tnpu_welford
triton_helpers.promote_to_tensor = _tnpu_promote_to_tensor
triton_helpers.is_floating = _tnpu_is_floating
triton_helpers.minimum = _tnpu_minimum
triton_helpers.maximum = _tnpu_maximum
triton_helpers.min2 = _tnpu_min2
triton_helpers.max2 = _tnpu_max2
'''


#: The mm and bmm templates' names for a row/column index, its bound, and the
#: block that walks it. Both templates spell them the same and both write the
#: wrap twice (once per branch of a contiguity test), which is why this is a
#: pattern rather than a line number.
_WRAP_TRIPLES = (("rm", "M", "BLOCK_M"), ("rn", "N", "BLOCK_N"))


def _literal_int(body, name):
    """`name`'s value if the kernel assigns it an integer literal, else None.

    Inductor renders M, N, K and the blocks into the BODY of the kernel rather
    than passing them (`M = 512`, `BLOCK_M : tl.constexpr = 256`), so the
    divisibility question can be answered from the source alone -- no need to
    thread the config down to here, and a dynamic shape simply has no literal
    and gets no rewrite.
    """
    m = re.search(rf"^\s*{name}\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)\s*$",
                  body, re.M)
    return int(m.group(1)) if m else None


#: Names the loads are keyed on. Both templates call the operand pointers A and
#: B and both keep `rm` / `rn` live across the k-loop (they are rematerialized
#: only for the store, after it), so the mask can be built from them directly
#: rather than from whichever name the wrap was assigned to.
_ROW_MASK = "_tnpu_row_mask"
_COL_MASK = "_tnpu_col_mask"
_MASK_FOR = {"A": _ROW_MASK, "B": _COL_MASK}


#: `tl.load(A ...` in every spelling the two templates use. mm builds the
#: address at the load -- `tl.load(A + (xindex))` -- while bmm accumulates it
#: into the pointer and writes `tl.load(A)` or `tl.load(A, mask=..., other=0.)`.
#: Matching only the first silently skipped every bmm, which then kept its wrap
#: and its gather while the log said nothing.
def _load_re(ptr):
    return re.compile(rf"\btl\.load\({ptr}\b")


def _add_mask_to_loads(lines, ptr, mask_name):
    """Give every `tl.load(<ptr>...)` the bound `mask_name`. Count applied.

    A load either already carries the template's K mask -- `mask=a_mask,
    other=0.0`, emitted when EVEN_K is false -- or carries none at all. The
    first is widened with `&` and the second gets one; both end up reading 0
    where the tile runs past the operand.
    """
    rx = _load_re(ptr)
    n = 0
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not rx.search(stripped) or not stripped.endswith(")"):
            continue
        at = stripped.find("mask=")
        if at < 0:
            lines[i] = stripped[:-1] + f", mask={mask_name}, other=0.0)"
            n += 1
            continue
        # NOT A REGEX. The existing mask is an EXPRESSION and it contains both
        # of the characters a regex would stop at: `mask=rk[None, :] < k` has a
        # comma inside a subscript and a `)` in neither place a naive `[^,)]+`
        # expects, and matching that way produced
        #   mask=(rk[None) & _tnpu_row_mask, :] < k
        # which is syntactically valid Python and means nothing. The end of the
        # expression is the ` other=` keyword the templates always follow it
        # with, or the load's own closing paren.
        start = at + len("mask=")
        end = stripped.find(", other=", start)
        if end < 0:
            end = len(stripped) - 1
        expr = stripped[start:end]
        lines[i] = (stripped[:start] + f"({expr}) & {mask_name}"
                    + stripped[end:])
        n += 1
    return n


def clamp_instead_of_wrap(body, kernel_name=""):
    """Replace the mm/bmm templates' `rm % M` / `rn % N` with a load mask.

    WHAT THE WRAP IS FOR. `rm = pid_m * BLOCK_M + arange(BLOCK_M)` runs past M
    whenever BLOCK_M does not divide M, so the last program addresses rows that
    do not exist. The template folds them back to the top of the matrix with
    `% M` and lets the store's mask discard whatever they computed. On a GPU
    that is the right trade: no branch, no fault, and reading garbage is free.

    WHAT IT COSTS HERE. A modulo on a pointer index is not a stride, so the
    descriptor stops being one: triton-shared marks it (`static_shape != 0`) and
    tnpu lowers it as an INDIRECT transfer -- a per-element index tile with the
    wrapped dim's stride set to 0. Three bills follow.

      the index tile is real scratchpad, in i64, the shape of the operand --
      TWICE the bytes of an f32 tile, and both operands are wrapped, so a
      three-tile kernel stages five. Measured on a 512x512x512 addmm with a
      residual: 73760 bytes/lane against a 65536 budget, refused at link.

      a gather is element-by-element where a descriptor is a burst.

      and it is where the wrong numbers are. Measured on the SAME kernel with
      only this changed: 512x512x512 at BM/BN = 256 answers 110.93 with the wrap
      and 5.34e-05 without, at the same tile and the same 2x2 grid.

    THE ANSWER IS THE ONE THIS BACKEND ALREADY USES ON THE OTHER SIDE. The store
    in the very same kernel does not fold, it CLAMPS:

        mask = (idx_m < M) & (idx_n < N)
        tl.store(out_ptr0 + xindex, acc, mask)

    which lowers to `tts.store`'s `static_mask_dims` and then to a transfer with
    `masked_axes` / `masked_fill = 0`. PyTorchSim's MLIR route answers the same
    question the same way on BOTH sides -- `def_dma_op` clamps each tile dim to
    the real DRAM extent and zero-fills past it, and its gemm template contains
    no modulo at all. The Triton route was the asymmetric one: clamping stores
    and folding loads.

    So the wrap goes and the loads get the matching bound. Out-of-range rows
    read 0 instead of a folded row, contribute 0 to the dot product, and land in
    output rows the store's mask already discards -- the same values are kept,
    by construction.

    A DIVIDING BLOCK NEEDS NEITHER. `gemm_combination_mapping` picks tiles from
    the DIVISORS of the shape, so rm runs 0..M-1 and both the wrap and the mask
    are dead; the wrap is dropped and no mask is added, which is what keeps the
    common case a bare descriptor.

    Torch's template already reasons this way about K -- `EVEN_K` skips the K
    mask -- and simply does not for M and N. This supplies the missing half.
    """
    lines = body.splitlines()
    text = "\n".join(lines)
    needs = {}
    for idx, dim, block in _WRAP_TRIPLES:
        d, b = _literal_int(text, dim), _literal_int(text, block)
        if d is None or not b:
            continue                    # dynamic shape: leave the wrap alone
        needs[idx] = bool(d % b)        # a tail exists -> the mask is load-bearing

    if not needs:
        return body

    # A DIVIDING BLOCK NEEDS NO LOAD AT ALL. Only the tail case has to find and
    # bound the loads, so the anchor and the load patterns are looked for ONLY
    # when some axis asks for a mask -- otherwise a template this does not
    # recognise still gets its dead wrap removed.
    if any(needs.values()):
        # The masks go right after offs_k / rk, which both templates emit after
        # rm/rn and before the k-loop, so every load is in their scope. mm calls
        # it offs_k and bmm calls it rk.
        anchor = next((i for i, l in enumerate(lines)
                       if l.strip().startswith(("offs_k = tl.arange(",
                                                "rk = tl.arange("))), None)
        if anchor is None or not any(_load_re(p).search(text) for p in _MASK_FOR):
            logger.warning(
                "[triton-npu] %s: a block does not divide its dimension and "
                "this does not recognise the loads to bound; leaving the wrap",
                kernel_name or "kernel")
            return body

        pad = " " * (len(lines[anchor]) - len(lines[anchor].lstrip()))
        inserted, applied = [], {}
        for idx, dim, _block in _WRAP_TRIPLES:
            if not needs.get(idx):
                continue
            name = _ROW_MASK if idx == "rm" else _COL_MASK
            slice_ = "[:, None]" if idx == "rm" else "[None, :]"
            inserted.append(f"{pad}{name} = {idx}{slice_} < {dim}")
        lines[anchor + 1:anchor + 1] = inserted
        for ptr, name in _MASK_FOR.items():
            if name in "".join(inserted):
                applied[ptr] = _add_mask_to_loads(lines, ptr, name)
        # A mask nobody could attach would leave the load unbounded once the
        # wrap is gone, so the wrap stays and this whole rewrite is given back.
        if any(v == 0 for v in applied.values()):
            logger.warning(
                "[triton-npu] %s: could not attach a bound to every load; "
                "leaving the wrap in place", kernel_name or "kernel")
            return body

    body = "\n".join(lines)
    for idx, dim, block in _WRAP_TRIPLES:
        if idx not in needs:
            continue
        body, n = re.subn(rf"\b{idx}\s*%\s*{dim}\b", idx, body)
        if n:
            logger.info(
                "[triton-npu] %s: replaced %d `%s %% %s` with %s, so the "
                "operand stays a descriptor instead of becoming a gather",
                kernel_name or "kernel", n, idx, dim,
                "a load bound" if needs[idx] else "nothing (the block divides)")
    return body


def strip_for_tnpu(src):
    """Remove everything the torch-free tnpu venv cannot import.

    Drops torch/inductor imports and the `@triton_heuristics.*(...)` decorator
    (keeping `@triton.jit`), then re-adds the two imports the kernel body needs.

    Raises SpecIncomplete if the kernel still calls into `triton_helpers`: that
    module lives in torch, so it has to be vendored into the tnpu venv before
    such a kernel can compile. Failing here names the missing helper; letting it
    through fails as a bare NameError inside tnpu's stage-1 worker instead.
    """
    lines = src.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if _HEURISTIC_RE.match(line.strip()) or _HEURISTIC_RE.match(line):
            # skip the whole decorator call, up to (not including) @triton.jit
            while i < len(lines) and lines[i].strip() != "@triton.jit":
                i += 1
            continue
        if _DROP_IMPORT_RE.match(line) or _DROP_CALL_RE.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    body = "\n".join(out)

    used = sorted(set(_HELPER_USE_RE.findall(body)))
    unvendored = [h for h in used if h not in _VENDORED_HELPERS]
    if unvendored:
        raise SpecIncomplete(
            f"kernel uses triton_helpers.{{{','.join(unvendored)}}}, which lives "
            f"in torch and the tnpu venv has no torch. Add it to _HELPERS_SRC if "
            f"it is pure triton, or lower it another way if it is not.")

    # The generated source already imports triton itself; only add what a
    # stripped module might be missing.
    prefix = ""
    if "import triton.language as tl" not in body:
        prefix = "import triton\nimport triton.language as tl\n\n"
    # tl_math is triton's own, re-exported through triton_helpers; the dropped
    # torch import took it with it.
    if re.search(r"\btl_math\.", body):
        prefix += "from triton.language import math as tl_math\n"

    # Same story as tl_math: Inductor reaches libdevice through torch, and the
    # dropped import took it with it. The name has to be bound to triton's OWN
    # copy -- libdevice members are @core.extern stubs, and a backend binds them
    # via get_module_map, which triton_shared now does. Leave the name unbound
    # and a call returns None, failing as "cannot convert None of type NoneType"
    # inside stage 1 without ever naming libdevice.
    if re.search(r"\blibdevice\.", body):
        prefix += "from triton.language.extra.cuda import libdevice\n"
    # Emitted into the spec rather than imported: the tnpu venv has no torch to
    # import from, and a module dropped beside the spec would depend on how
    # tnpu's stage-1 worker happens to set sys.path.
    if used:
        prefix += _HELPERS_SRC
    return prefix + body


def scalar_args(meta):
    """User scalar parameters, in kernel order, as [(name, c_type, value)].

    triton-shared keeps these in the lowered signature ahead of its own six
    grid/pid arguments, so the wrapper must pass them or every later argument
    lands one slot early -- pidX then reads pidY and only program 0 runs.
    """
    numels = meta["numels"]
    out = []
    for name, token in meta["signature"].items():
        if token.startswith("*") or token == "constexpr":
            continue
        ctype = _C_TYPE.get(token)
        if ctype is None:
            raise SpecIncomplete(
                f"{meta['kernel_name']}: scalar '{name}' has type {token!r}, "
                f"which has no C mapping in _C_TYPE")
        if numels.get(name) is None:
            raise SpecIncomplete(
                f"{meta['kernel_name']}: no value for scalar '{name}' -- "
                f"collect_meta resolves these from kernel.numels")
        out.append((name, ctype, int(numels[name])))
    return out


def grid_of(meta):
    """Launch grid, from the numels and the pinned block sizes, outermost first.

    Also read by the timing path, which needs the same extents to enumerate the
    work-items -- so it lives here rather than being recomputed per consumer.
    """
    numels = meta["numels"]
    cfg = meta.get("fixed_config") or {}
    axes = parallel_axes(numels)
    if not axes:
        raise SpecIncomplete(
            f"{meta['kernel_name']} has no parallel iteration axis to grid over")

    grid = []
    for prefix in axes:
        n, block = numels.get(f"{prefix}numel"), cfg.get(_block_name(prefix))
        if n is None or not block:
            raise SpecIncomplete(
                f"cannot compute the grid for {meta['kernel_name']} axis "
                f"'{prefix}': {prefix}numel={n!r}, {_block_name(prefix)}={block!r}. "
                f"Inductor defers the grid to triton_heuristics at runtime; this "
                f"route needs it statically (see fixed_config_for).")
        grid.append(int(math.ceil(n / block)))
    return tuple(grid)


def _template_grid(kernel):
    """A template kernel's grid, from the template rather than from the numels.

    mm and conv do not come from Inductor's pointwise codegen: their source is a
    jinja template with its own BLOCK_M/N/K baked in as literals, and their grid
    is over OUTPUT TILES -- select_algorithm.TritonTemplateKernel.call_kernel
    emits `*grid_fn(*call_sizes, meta)`. Nothing about that is derivable from
    xnumel and XBLOCK.

    Deriving it that way anyway is what a pointwise formula does to an mm: an
    18432-element output at XBLOCK 128 becomes grid 144, when the template wants
    cdiv(M, 32) * cdiv(N, 32). The kernel then runs the wrong number of programs
    over tiles it never asked for, and nothing about the shapes disagrees loudly
    enough to notice.

    Returns None for an ordinary kernel, which is every kernel that HAS numels.
    """
    grid_fn = getattr(kernel, "grid_fn", None)
    sizes = getattr(kernel, "call_sizes", None)
    if grid_fn is None or sizes is None:
        return None
    try:
        vals = [int(V.graph.sizevars.size_hint(s)) for s in sizes]
        g = grid_fn(*vals, dict(getattr(kernel, "meta", None) or {}))
    except Exception as e:  # noqa: BLE001 - reported by grid_xyz, with the cause
        return {"error": f"{type(e).__name__}: {e}"}
    return [int(x) for x in g]


def grid_xyz(meta):
    """The same grid in tnpu's order: (gridX, gridY, gridZ).

    tnpu's spec.grid is positional and X-FIRST -- wrapper._grid3 reads
    `g[0], g[1], g[2]` as gx, gy, gz and emits the loop nest in that order --
    while grid_of is OUTERMOST-first (z, y, x) to match Inductor's axes. For a
    1D kernel the two orders are the same tuple, which is why every kernel on
    this route agreed until a second axis appeared.

    Handed the outermost-first tuple, tnpu reads gridY as gridX: the inner axis
    then runs for as many steps as the outer one needed, and a transpose comes
    back with only its first XBLOCK columns written and the rest left at zero.
    Silently wrong output, not a crash, so it is built here by axis NAME.
    """

    tg = meta.get("template_grid")
    if tg is not None:
        if isinstance(tg, dict):
            raise SpecIncomplete(
                f"{meta['kernel_name']} is a template kernel and its own grid "
                f"function could not be evaluated ({tg['error']}); the pointwise "
                f"formula would silently give the wrong number of programs")
        return tuple(list(tg)[:3] + [1, 1, 1])[:3]

    extents = dict(zip(parallel_axes(meta["numels"]), grid_of(meta)))
    return tuple(extents.get(p, 1) for p in ("x", "y", "z"))


SPEC_TEMPLATE = '''\
"""Generated by PyTorchSimFrontend/triton_backend/kernel_spec.py -- do not edit.

Inductor kernel {kernel_name!r}, rewritten for the tnpu pipeline: the
triton_heuristics autotuner decorator is stripped and its block sizes are pinned
as constexprs, so the launch shape is static. See kernel_spec.py for why.
"""
import importlib.util
import os
import sys

sys.path.insert(0, {tnpu_dir!r})
from tnpu.spec import KernelSpec, Arg  # noqa: E402

#: The rewritten Triton source, beside this file. It must be a REAL file on
#: disk, not an exec'd string: triton's @jit reads the function back with
#: inspect.getsourcefile and rejects anything else ("@jit functions should be
#: defined in a Python file").
TRITON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           {triton_module!r})


def kernel():
    spec = importlib.util.spec_from_file_location(
        {kernel_name!r} + "_triton", TRITON_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, {kernel_name!r})


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {{}}
{make_inputs_body}
    return out


def reference(inputs):
    # The Inductor route has no per-kernel torch reference: correctness is
    # checked at the graph level by the test that ran torch.compile. tnpu's
    # stage 7 is therefore not meaningful here and the pipeline is driven to
    # stage 6 (spike) instead.
    return {{}}


SPEC = KernelSpec(
    name={kernel_name!r},
    kernel=kernel,
    signature={signature!r},
    constexprs={constexprs!r},
    args=[
{args_body}
    ],
    grid={grid!r},
    reference=reference,
    make_inputs=make_inputs,
    extra={{"scalar_args": {scalar_decls!r},
           "scalar_values": {scalar_values!r}}},
    notes="generated from Inductor triton codegen",
)
'''


def write_spec_file(src_code, meta, path, tnpu_dir):
    """Write a tnpu kernel file for this Inductor kernel. Returns `path`."""
    missing = [a["name"] for a in meta["args"] if not a["dtype"] or not a["numel"]]
    if missing:
        raise SpecIncomplete(
            f"{meta['kernel_name']}: no dtype/numel for {missing} -- "
            f"collect_meta could not resolve them from V.graph")

    signature = dict(meta["signature"])
    constexprs = dict(meta["constants"])
    for k, v in (meta.get("fixed_config") or {}).items():
        if k not in signature:
            # Inductor already fixed this one in the kernel BODY rather than
            # taking it as a parameter -- a persistent reduction does that with
            # R0_BLOCK. Passing it would not match the signature, and there is
            # nothing left for us to choose.
            continue
        if v is None:
            raise SpecIncomplete(
                f"{meta['kernel_name']}: block size {k} is unset "
                f"(fixed_config_for leaves reduction blocks unset on purpose)")
        constexprs[k] = v
        signature[k] = "constexpr"

    args_body = "\n".join(
        f"        Arg({a['name']!r}, {a['role']!r}, {a['dtype']!r}, ({a['numel']},)),"
        for a in meta["args"])
    make_inputs_body = "\n".join(
        f"    out[{a['name']!r}] = torch.randn({a['numel']}, generator=g)"
        f".to(torch.{a['dtype']})"
        for a in meta["args"] if a["role"] in ("in", "inout")) or "    pass"

    triton_module = f"{meta['kernel_name']}_triton.py"
    with open(os.path.join(os.path.dirname(path), triton_module), "w") as f:
        f.write(clamp_instead_of_wrap(strip_for_tnpu(src_code),
                                      meta["kernel_name"]))

    scalars = scalar_args(meta)
    text = SPEC_TEMPLATE.format(
        kernel_name=meta["kernel_name"],
        tnpu_dir=tnpu_dir,
        triton_module=triton_module,
        signature=signature,
        constexprs=constexprs,
        args_body=args_body,
        make_inputs_body=make_inputs_body,
        grid=grid_xyz(meta),
        scalar_decls=[(n, c) for n, c, _ in scalars],
        scalar_values={n: v for n, _, v in scalars},
    )
    with open(path, "w") as f:
        f.write(text)
    return path
