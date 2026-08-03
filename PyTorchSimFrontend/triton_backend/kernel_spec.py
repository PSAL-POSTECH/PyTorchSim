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

from . import helpers_shim

#: Triton signature token -> (torch dtype name, bytes). Only the dtypes
#: tnpu/wrapper.py can round-trip through .raw files.
_DTYPE = {
    "*fp32": "float32", "*fp16": "float16", "*bf16": "bfloat16",
    "*i64": "int64", "*i32": "int32", "*i8": "int8", "*i1": "bool",
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
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": fixed_config_for(kernel),
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


def fixed_config_for(kernel):
    """Block sizes pinned at codegen time.

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

    axes = parallel_axes(getattr(kernel, "numels", None) or {})
    cfg = {_block_name(p): (lanes if i == 0 else 1) for i, p in enumerate(axes)}
    if len(axes) > 1:
        # Loud, because the shape is correct but pathological: an inner block of
        # 1 makes every work-item move a strided column. Fine for getting a
        # multi-axis kernel through the route, misleading to benchmark.
        extension_config.setup_logger().warning(
            "[triton-npu] %s tiles over %s; inner blocks pinned to 1, which is "
            "correct but not a tiling worth measuring",
            getattr(kernel, "kernel_name", "kernel"), axes)
    cfg.setdefault("XBLOCK", lanes)         # a kernel with no tiling info still has x
    if getattr(kernel, "inside_reduction", False):
        # A reduction block is NOT free to be the lane count: the reduced axis
        # has to stay inside a lane (see triton-npu kernels/reduce.py). Left
        # unset on purpose so the reduction path fails loudly rather than
        # silently picking a layout the hardware cannot execute.
        cfg["R0_BLOCK"] = None
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

    prefix = (
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.language import math as tl_math\n"
        "from triton.language.extra import libdevice\n"
        f"from {helpers_shim.PACKAGE} import triton_helpers\n\n"
    )
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
# This directory too: the kernel is loaded by path, so a sibling package
# (tnpu_helpers) would not otherwise be importable from it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    stripped = strip_for_tnpu(src_code)
    if helpers_shim.PACKAGE in stripped:
        helpers_shim.write_package(os.path.dirname(path))
    with open(os.path.join(os.path.dirname(path), triton_module), "w") as f:
        f.write(stripped)

    scalars = scalar_args(meta)
    text = SPEC_TEMPLATE.format(
        kernel_name=meta["kernel_name"],
        tnpu_dir=tnpu_dir,
        triton_module=triton_module,
        signature=signature,
        constexprs=constexprs,
        args_body=args_body,
        make_inputs_body=make_inputs_body,
        grid=grid_of(meta),
        scalar_decls=[(n, c) for n, c, _ in scalars],
        scalar_values={n: v for n, _, v in scalars},
    )
    with open(path, "w") as f:
        f.write(text)
    return path
