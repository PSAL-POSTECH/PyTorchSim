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
    """How many elements of STORAGE an Inductor buffer spans, or None.

    NOT the product of its shape, which is what this used to be. The kernel
    addresses with the buffer's STRIDES -- Inductor bakes them into the source
    as constants -- so the buffer it is handed has to be as long as those
    strides reach, and a layout whose strides leave gaps reaches further than
    its element count:

        buf18 = empty_strided((1, 12, 197, 197), (465792, 38816, 197, 1))
                   197 * 197 = 38809 elements per head, at a PITCH of 38816

    Seven elements of hole per head, twelve heads: 465792 of storage against
    465708 of shape. Sized by the shape, the last head runs 77 elements past
    the end of the buffer the harness allocated, and those 77 are simply lost on
    the way back -- which is what ViT's first functional-verify failure was.
    Every element of `buf18` read as if the heads were packed, 416959 of 465708
    over tolerance, while the kernel had in fact computed the softmax correctly
    (0.0318 max error over the padded reading, and all of that in the truncated
    tail).

    `1 + sum((size - 1) * stride)` is the last addressable element, which is the
    definition that holds for a permuted layout as well as a padded one -- a
    channels-last buffer has no gaps and comes out at the product, as before.
    """
    try:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None
        layout = buf.get_layout()
        hint = V.graph.sizevars.size_hint
        size = [int(hint(s)) for s in layout.size]
        if any(s <= 0 for s in size):
            return 0
        try:
            stride = [int(hint(s)) for s in layout.stride]
        except (AttributeError, TypeError):
            # A layout with no strides to read: fall back to the shape, which
            # is what this function always did and is right whenever the buffer
            # is contiguous.
            n = 1
            for s in size:
                n *= s
            return n
        offset = int(hint(getattr(layout, "offset", 0)))
        return offset + 1 + sum((s - 1) * t for s, t in zip(size, stride))
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
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": fixed_config_for(kernel, numels, args),
        "template_grid": _template_grid(kernel),
    }


def _template_grid(kernel):
    """A TEMPLATE kernel's launch grid as (gridX, gridY, gridZ), else None.

    A template kernel (mm, conv) does not walk the output the way a pointwise
    kernel does. It tiles with its own BLOCK_M/BLOCK_N and reads
    `tl.program_id(0..2)` directly, so the grid is the template's to state --
    Inductor passes it as three extra launcher arguments rather than deriving it
    (`FixedGrid.setup_grid_as_args`). The `numels` such a kernel reports describe
    the output tensor, not its iteration space, so ceil(numel / XBLOCK) is not a
    grid at all: for ResNet's first conv it gave 6272 where the template asks for
    196, and the surplus programs index tiles past the end of every operand.

    Asked the same way Inductor asks it for its own benchmark harness
    (`TritonTemplateKernel.kernel_benchmark_extra_args`), so the grid here is the
    grid the launcher would have passed.
    """
    grid_fn = getattr(kernel, "grid_fn", None)
    call_sizes = getattr(kernel, "call_sizes", None)
    if grid_fn is None or call_sizes is None:
        return None                       # not a template; the numels are real
    grid = grid_fn(*V.graph.sizevars.size_hints(call_sizes), kernel.meta)
    try:
        extents = tuple(int(g) for g in grid)
    except TypeError:
        extents = ()
    if len(extents) != 3 or any(g < 1 for g in extents):
        raise SpecIncomplete(
            f"{getattr(kernel, 'kernel_name', kernel)}: template grid_fn returned "
            f"{grid!r}; this route needs three static positive extents. Refusing "
            f"rather than falling back to the numels, which describe the output "
            f"tensor and not this kernel's iteration space.")
    return extents


#: Parallel iteration prefixes, OUTERMOST first. Inductor's `x` is the
#: contiguous axis, so it is innermost; `r*` prefixes are reductions, looped
#: inside the kernel rather than spread over the grid (prefix_is_reduction).
_PARALLEL_PREFIXES = ("z", "y", "x")


def _block_name(prefix):
    return f"{prefix.upper()}BLOCK"


def parallel_axes(numels):
    """Grid axes this kernel uses, outermost first."""
    return [p for p in _PARALLEL_PREFIXES if f"{p}numel" in numels]


def launch_axes(meta, numels=None):
    """The axes the LAUNCH spreads programs over, outermost first.

    Same question as `parallel_axes`, asked of the whole kernel rather than of
    its numels, because a template kernel's numels do not answer it: it declares
    its grid outright. Every consumer that pairs an axis with an extent -- the
    spec's grid, the trace's shape file, the WorkItem's program-id arguments --
    must ask this one, or two of them disagree about which slot is which.
    """
    grid = meta.get("template_grid")
    if grid is None:
        return parallel_axes(meta["numels"] if numels is None else numels)
    # ALL THREE, including a slot whose extent is 1. A pointwise kernel reads
    # exactly the program ids its numels name, but a template reads
    # `tl.program_id(0..2)` from its own text no matter what the grid says --
    # ResNet's first conv multiplies pidY by BLOCK_N even though gridY is 1. An
    # axis left undeclared is a kernel argument nothing accounts for, and the
    # trace producer stops at it ("kernel arg still used after build_skeleton").
    # A declared axis of extent 1 just runs its loop once.
    return list(_PARALLEL_PREFIXES)


def launch_extents(meta, numels=None):
    """The extents of `launch_axes`, in the same order.

    Kept beside the axes rather than recomputed per consumer: the pairing is the
    part that has been got wrong, not either half on its own.
    """
    grid = meta.get("template_grid")
    if grid is None:
        return grid_of(meta if numels is None else {**meta, "numels": numels})
    by_axis = dict(zip(("x", "y", "z"), grid))
    return tuple(by_axis[p] for p in launch_axes(meta, numels))


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
#:
#: IT IS AN OPENING BID NOW, NOT AN ANSWER, and the sentence above says why it
#: could never have been one: the count is a property of the LOWERING, which
#: does not exist until this number has already been used. ViT's first LayerNorm
#: -- fused with a patch convolution, an addmm and a transpose -- keeps 41
#: scratchpad globals live, and no constant that serves that kernel is tolerable
#: for an ordinary reduction (it would cost three quarters of the tile). So
#: codecache._shrink_reduction_blocks corrects it instead: tnpu measures the
#: real usage, says by how much it is over, and the kernel is recompiled with a
#: block divided by that ratio. Guessing low now costs one recompile rather than
#: a failure, which is what makes 12 an acceptable guess rather than a bet.
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

    Tile dim 0 is the one `bank_vectorize` spreads over the lanes, so the
    OUTERMOST axis gets the lane count -- a per-lane depth of 1, the shape every
    tnpu baseline runs. The remaining axes get 1, which leaves the tile exactly
    that verified shape and lets the grid cover the rest. It is conservative
    rather than fast; choosing real tile sizes is the block-size policy gap in
    README, not something to guess at here.
    """
    from PyTorchSimFrontend import extension_config
    lanes = int(extension_config.vpu_num_lanes)
    vlen = int(extension_config.vpu_vector_length_bits)
    per_vector = max(1, vlen // _element_bits(args))

    axes = parallel_axes(numels)
    cfg = {}
    for i, p in enumerate(axes):
        if i == 0:
            # Tile dim 0. The MVIN DMA scatters it across the lanes, and it is
            # also the systolic array's side (tnpu config.py: sa_dim = n_vu), so
            # the lane count is the one value that satisfies both -- which is
            # why matmul needs this read from the config rather than assumed.
            #
            # WHICH LETTER dim 0 IS DEPENDS ON RANK, so it is indexed rather
            # than named. Inductor gives the outermost axis the [:, None] slot,
            # so a 2D kernel puts y there and a 1D one has only x. Confirmed in
            # 04-custom.mlir: the 2D transpose stages memref<128x8xf32, 1> with
            # vlane_split_axis = 0 under YBLOCK=128, XBLOCK=8, and 1D add stages
            # memref<128xf32, 1>, same axis, under XBLOCK=128.
            cfg[_block_name(p)] = lanes
        elif i == len(axes) - 1:
            # Innermost, so contiguous (tile_stride ends in 1): give each lane
            # one full vector register. 8 fp32 at vlen 256 is exactly that.
            # Pinned to 1 before, which is why a multi-axis work-item moved a
            # strided column -- correct but nothing worth measuring.
            cfg[_block_name(p)] = per_vector
        else:
            cfg[_block_name(p)] = 1
    cfg.setdefault("XBLOCK", lanes)         # a kernel with no tiling info still has x
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
        lane_bytes = int(os.environ.get("TNPU_SPAD_SIZE", str(64 * 1024)), 0)
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
                     "welford_reduce", "welford_combine", "welford",
                     "sort_with_index"}

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

# --- bitonic sort, for top-k -------------------------------------------------
# `_log2` is TRITON'S OWN (triton.language.standard), not torch's -- torch
# imports it from there too, through triton_compat. So the chain below reaches
# nothing outside triton once is_floating is the vendored one.
from triton.language.standard import _log2 as _tnpu_log2

@triton.jit
def _tnpu_compare_and_swap_with_index(
    x, idxs, rnumel, flip,
    i: tl.constexpr, n_dims: tl.constexpr,
    stable: tl.constexpr, descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    y = tl.reshape(x, shape)
    iy = y.to(idtype, bitcast=True)
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)
    left_mask = (1 - right_mask).to(idtype)
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1).to(idtype)[:, None, :], shape)
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1).to(idtype)[:, None, :], shape)
    ileft = tl.reshape(ileft, x.shape)
    iright = tl.reshape(iright, x.shape)
    left = ileft.to(x.dtype, bitcast=True)
    right = iright.to(x.dtype, bitcast=True)

    y_idx = tl.reshape(idxs, shape)
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    if rnumel is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)
        right_valid_mask = tl.full(x.shape, True, tl.int1)
    else:
        left_valid_mask = left_idx < rnumel
        right_valid_mask = right_idx < rnumel

    ix = x.to(idtype, bitcast=True)

    # sort treats nan as the higher value, and comparisons with nan are always
    # False -- so the isnan terms are load-bearing, exactly as in _tnpu_maximum.
    left_isnan = left != left
    right_isnan = right != right

    if descending:
        cond = left < right
        if _tnpu_is_floating(left):
            if not stable:
                cond = cond | right_isnan
            else:
                cond = cond | (right_isnan & (~left_isnan))
    else:
        cond = left > right
        if _tnpu_is_floating(left):
            if not stable:
                cond = cond | left_isnan
            else:
                cond = cond | (left_isnan & (~right_isnan))

    if stable:
        eq = left == right
        if _tnpu_is_floating(left):
            eq = eq | (left_isnan & right_isnan)
        cond = cond | (eq & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )
    cond = (cond ^ flip).to(tl.int1)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))

    return ret.to(x.dtype, bitcast=True), new_idxs

@triton.jit
def _tnpu_bitonic_merge_with_index(
    x, idxs, rnumel,
    stage: tl.constexpr, alternating: tl.constexpr, n_dims: tl.constexpr,
    stable: tl.constexpr, descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    for i in tl.static_range(stage):
        x, idxs = _tnpu_compare_and_swap_with_index(
            x, idxs, rnumel, flip, i + (n_dims - stage), n_dims, stable, descending
        )
    return x, idxs

@triton.jit
def _tnpu_sort_with_index(
    x, idxs, rnumel,
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    x, idxs = tl.broadcast(x, idxs)
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    n_dims: tl.constexpr = _tnpu_log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, idxs = _tnpu_bitonic_merge_with_index(
            x, idxs, rnumel, i,
            alternating=i < n_dims, n_dims=n_dims,
            stable=stable, descending=descending,
        )
    return x, idxs

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
triton_helpers.sort_with_index = _tnpu_sort_with_index
'''


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

    A template kernel states its own grid and this derivation does not apply to
    it; see `_template_grid`.
    """
    if meta.get("template_grid") is not None:
        return launch_extents(meta)

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
    extents = dict(zip(launch_axes(meta), launch_extents(meta)))
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
        f.write(strip_for_tnpu(src_code))

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
