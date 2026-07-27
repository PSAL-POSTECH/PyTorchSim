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
    "*fp32": "float32", "*fp16": "float16", "*bf16": "bfloat16",
    "*i64": "int64", "*i32": "int32", "*i8": "int8", "*i1": "bool",
    "fp32": "float32", "i32": "int32", "i64": "int64",
}


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


def fixed_config_for(kernel):
    """Block sizes pinned at codegen time.

    tnpu compiles ONE binary per kernel and the C wrapper walks the grid as a
    sequential loop, so there is no autotuner to choose XBLOCK later and no
    runtime `grid=` callable. Fixing it here is what makes the launch shape
    static.

    The lane count is the natural default: `bank_vectorize` distributes tile
    dim 0 across the lanes, and a block equal to the lane count gives a per-lane
    depth of 1 -- the case every tnpu baseline runs today.
    """
    from PyTorchSimFrontend import extension_config
    lanes = int(extension_config.vpu_num_lanes)
    cfg = {"XBLOCK": lanes}
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
#: Anything else from triton_helpers is a real dependency -- maximum/minimum/
#: promote_to_tensor and friends, which reductions and clamps use constantly.
_HELPER_USE_RE = re.compile(r"\btriton_helpers\.(\w+)")


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
    if used:
        raise SpecIncomplete(
            f"kernel uses triton_helpers.{{{','.join(used)}}}, which lives in "
            f"torch and the tnpu venv has no torch. Vendor a minimal "
            f"triton_helpers into the tnpu venv (or into TRITON_SRC) before this "
            f"kernel can compile.")

    # The generated source already imports triton itself; only add what a
    # stripped module might be missing.
    prefix = ""
    if "import triton.language as tl" not in body:
        prefix = "import triton\nimport triton.language as tl\n\n"
    return prefix + body


def grid_of(meta):
    """Launch grid, from the numels and the pinned block sizes.

    Also read by the timing path, which needs the same extents to enumerate the
    work-items -- so it lives here rather than being recomputed per consumer.
    """
    x = meta["numels"].get("xnumel")
    xblock = (meta.get("fixed_config") or {}).get("XBLOCK")
    if x is None or not xblock:
        raise SpecIncomplete(
            f"cannot compute the grid for {meta['kernel_name']}: "
            f"xnumel={x!r}, XBLOCK={xblock!r}. Inductor defers the grid to "
            f"triton_heuristics at runtime; this route needs it statically "
            f"(see fixed_config_for).")
    return (int(math.ceil(x / xblock)),)


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

    text = SPEC_TEMPLATE.format(
        kernel_name=meta["kernel_name"],
        tnpu_dir=tnpu_dir,
        triton_module=triton_module,
        signature=signature,
        constexprs=constexprs,
        args_body=args_body,
        make_inputs_body=make_inputs_body,
        grid=grid_of(meta),
    )
    with open(path, "w") as f:
        f.write(text)
    return path
