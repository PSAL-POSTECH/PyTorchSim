"""Inductor kernel  ->  tnpu KernelSpec.

`collect_meta` runs at codegen time, while `V.graph` still exists, and
`write_spec_file` turns the Triton source plus that metadata into a file
`tnpu.spec.load_spec` can load.

The source is rewritten because Inductor's `@triton_heuristics.pointwise`
imports torch, which the tnpu venv does not have, and picks the block size at
runtime, which tnpu needs as a constexpr.
"""

import math
import os
import re

from torch._inductor.virtualized import V

from . import helpers_shim

#: Triton signature token -> torch dtype name; the full set Inductor's
#: `_type_of` (torch/_inductor/codegen/triton_utils.py) can emit.
_DTYPE = {
    "*fp64": "float64", "*fp32": "float32", "*fp16": "float16",
    "*bf16": "bfloat16",
    "*i64": "int64", "*i32": "int32", "*i16": "int16", "*i8": "int8",
    "*i1": "bool",
    "*u64": "uint64", "*u32": "uint32", "*u16": "uint16", "*u8": "uint8",
    "fp64": "float64", "fp32": "float32", "fp16": "float16",
    "bf16": "bfloat16",
    "i64": "int64", "i32": "int32", "i16": "int16", "i8": "int8", "i1": "bool",
    "u64": "uint64", "u32": "uint32", "u16": "uint16", "u8": "uint8",
}


#: Triton scalar token -> C type, for the wrapper's kernel declaration.
_C_TYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}


class SpecIncomplete(RuntimeError):
    """Metadata tnpu requires that this kernel did not provide, named rather
    than left to fail deeper in the pipeline.
    """


# ---------------------------------------------------------------------------
# 1. codegen-time metadata capture
# ---------------------------------------------------------------------------
def _buffer_layout(name):
    """(numel, size, stride) of an Inductor buffer; Nones if unresolvable.
    Inductor allocates outputs empty_strided and indexes them by stride, so a
    launch assuming contiguous writes to the wrong places.
    """
    try:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None, None, None
        layout = buf.get_layout()
        hint = V.graph.sizevars.size_hint
        size = [int(hint(s)) for s in layout.size]
        stride = [int(hint(s)) for s in layout.stride]
        n = 1
        for s in size:
            n *= s
        return n, size, stride
    except Exception:  # noqa: BLE001 - best effort; caller reports it as missing
        return None, None, None


def _roles(kernel):
    """arg name -> 'in' | 'out' | 'inout', from the kernel's buffer tables."""
    out = {}
    for buf, arg in getattr(kernel.args, "input_buffers", {}).items():
        out[arg] = ("in", buf)
    for buf, arg in getattr(kernel.args, "output_buffers", {}).items():
        # Mutated, not produced: unwritten elements must survive, so seed it.
        role = "inout" if buf in getattr(V.graph, "graph_inputs", {}) else "out"
        out[arg] = (role, buf)
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
        numel, size, stride = _buffer_layout(buf) if buf else (None, None, None)
        args.append({
            "name": name,
            "role": role,
            "buffer": buf,
            "dtype": _DTYPE.get(signature.get(name, ""), None),
            "numel": numel,
            "size": size,
            "stride": stride,
        })

    # kernel.numels is keyed by iteration-space prefix ('x', 'y', 'r0'), not by
    # xnumel/rnumel attributes. The grid is computed from these.
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


#: Parallel iteration prefixes, outermost first -- Inductor's `x` is the
#: contiguous axis. `r*` are reductions, looped inside the kernel.
_PARALLEL_PREFIXES = ("z", "y", "x")


def _block_name(prefix):
    return f"{prefix.upper()}BLOCK"


def parallel_axes(numels):
    """Grid axes this kernel uses, outermost first. For tile-shape decisions."""
    return [p for p in _PARALLEL_PREFIXES if f"{p}numel" in numels]


def pid_axes(numels):
    """The same axes in program-id order: pid 0 is x, whatever the tiling.
    Every grid tuple is in this order; tnpu reads spec.grid positionally.
    """
    return list(reversed(parallel_axes(numels)))


def fixed_config_for(kernel):
    """Block sizes pinned at codegen time; tnpu has no autotuner to pick them
    later. Tile dim 0 is the one bank_vectorize spreads over the lanes, so the
    outermost axis gets the lane count and the rest get 1 -- conservative, and
    the block-size policy gap in README.
    """
    from PyTorchSimFrontend import extension_config
    try:
        lanes = int(extension_config.vpu_num_lanes)
    except KeyError:
        raise SpecIncomplete(
            f"{extension_config.CONFIG_TOGSIM_CONFIG} has no vpu_num_lanes. "
            f"This route pins every block size to the lane count, so a config "
            f"without a VPU cannot describe a launch shape.") from None

    # parallel_axes wants collect_meta's "<prefix>numel" keys, not raw prefixes.
    axes = parallel_axes([f"{p}numel"
                          for p in (getattr(kernel, "numels", None) or {})])
    cfg = {_block_name(p): (lanes if i == 0 else 1) for i, p in enumerate(axes)}
    if len(axes) > 1:
        # Correct but pathological: an inner block of 1 moves a strided column
        # per work-item. Fine for coverage, misleading to benchmark.
        extension_config.setup_logger().warning(
            "[triton-npu] %s tiles over %s; inner blocks pinned to 1, which is "
            "correct but not a tiling worth measuring",
            getattr(kernel, "kernel_name", "kernel"), axes)
    cfg.setdefault("XBLOCK", lanes)         # a kernel with no tiling info still has x
    if getattr(kernel, "inside_reduction", False):
        # The whole reduced extent, so the kernel's r0 loop runs once.
        r0 = (getattr(kernel, "numels", None) or {}).get("r0_")
        try:
            cfg["R0_BLOCK"] = int(V.graph.sizevars.size_hint(r0))
        except Exception:  # noqa: BLE001 - dynamic; write_spec_file reports it
            cfg["R0_BLOCK"] = None
    return cfg


# ---------------------------------------------------------------------------
# 2. Triton source -> tnpu kernel file
# ---------------------------------------------------------------------------
_HEURISTIC_RE = re.compile(r"^@triton_heuristics\.")
_DROP_IMPORT_RE = re.compile(
    r"^\s*(import torch|from torch\b|from __future__|import __main__)")
#: GPU-only calls. set_driver_to_gpu picks a runtime we never launch through;
#: debug_barrier orders warps that do not exist here and reaches ttir as
#: ttg.barrier, which triton-shared-opt cannot parse.
_DROP_CALL_RE = re.compile(
    r"^\s*(triton_helpers\.set_driver_to_gpu|tl\.debug_barrier)\(\)")


def strip_for_tnpu(src):
    """Remove everything the torch-free tnpu venv cannot import: torch imports
    and the @triton_heuristics decorator, keeping @triton.jit. Raises
    SpecIncomplete naming any triton_helpers call not yet vendored, which would
    otherwise be a bare NameError inside tnpu's stage-1 worker.
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
    triton-shared keeps these ahead of its own grid/pid arguments, so omitting
    one shifts every later argument a slot early.
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
    """Launch grid, from the numels and the pinned block sizes, in pid order.

    Also read by the timing path, which needs the same extents to enumerate the
    work-items -- so it lives here rather than being recomputed per consumer.
    """
    numels = meta["numels"]
    cfg = meta.get("fixed_config") or {}
    axes = pid_axes(numels)
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
# The kernel is loaded by path, so a sibling package is not otherwise
# importable from it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnpu.spec import KernelSpec, Arg  # noqa: E402

#: The rewritten Triton source, beside this file. A real file, not an exec'd
#: string: @jit reads the function back with inspect.getsourcefile.
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
    # No per-kernel torch reference here -- correctness is checked at the graph
    # level -- so the pipeline stops at stage 6 rather than tnpu's verify.
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
            # Fixed in the kernel body rather than taken as a parameter (a
            # persistent reduction does this with R0_BLOCK); nothing to choose.
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
