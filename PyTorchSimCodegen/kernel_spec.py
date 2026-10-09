"""Inductor kernel -> the compiler KernelSpec.

`collect_meta` pulls what the compiler needs out of the Inductor kernel while `V.graph`
is still live; `write_spec_file` turns the Triton source plus that metadata into
a file the compiler's `spec.load_spec` can read. Launch shape is launch.py's, source
rewriting is source_rewrite.py's.
"""

import os
import re

from torch._inductor.virtualized import V

from PyTorchSimFrontend import config

from . import compiler_bridge  # noqa: F401
from . import layout, launch, source_rewrite
from .errors import SpecIncomplete

logger = config.setup_logger()

_DTYPE = {
    "*fp64": "float64", "*fp32": "float32", "*fp16": "float16",
    "*bf16": "bfloat16",
    # fp8 has no numpy or C scalar: it rides as uint8 and is viewed back at the
    # ends, the way bf16 rides as uint16. See functional.py's _np_dtype.
    "*fp8e5": "float8_e5m2", "*fp8e4nv": "float8_e4m3fn",
    "*i64": "int64", "*i32": "int32", "*i16": "int16", "*i8": "int8",
    "*u64": "uint64", "*u32": "uint32", "*u16": "uint16", "*u8": "uint8",
    "*i1": "bool",
    "fp32": "float32", "i32": "int32", "i64": "int64",
}

_C_TYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}


def _buffer_numel(name):
    """How many elements of STORAGE an Inductor buffer spans, or None.

    NOT the product of its shape: the kernel addresses with the strides, and a
    layout with gaps reaches further (ViT's buf18, 465792 against 465708).
    """
    try:
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None
        lay = buf.get_layout()
        hint = V.graph.sizevars.size_hint
        size = [int(hint(s)) for s in lay.size]
        if any(s <= 0 for s in size):
            return 0
        try:
            stride = [int(hint(s)) for s in lay.stride]
        except (AttributeError, TypeError):
            n = 1
            for s in size:
                n *= s
            return n
        return layout.storage_span(size, stride,
                                   int(hint(getattr(lay, "offset", 0))))
    except Exception:
        return None


def _roles(kernel):
    """arg name -> ('in'|'out'|'inout', buffer), from the kernel's tables.

    `inplace_buffers` IS NOT THE WHOLE OF `inout`: a kernel can mutate a buffer
    an EARLIER kernel produced, and `kernel.mutations` is what names that.
    """
    out = {}
    for buf, arg in getattr(kernel.args, "input_buffers", {}).items():
        out[arg] = ("in", buf)
    for buf, arg in getattr(kernel.args, "output_buffers", {}).items():
        out[arg] = ("out", buf)
    for buf, arg in getattr(kernel.args, "inplace_buffers", {}).items():
        name = getattr(arg, "inner_name", arg)
        out[name] = ("inout", buf)
    outputs = getattr(kernel.args, "output_buffers", {})
    for buf in getattr(kernel, "mutations", ()):
        arg = outputs.get(buf)
        name = getattr(arg, "inner_name", arg)
        if isinstance(name, str) and name in out:
            out[name] = ("inout", buf)
    return out


_STORE_RE = re.compile(r"\btl\.(store|atomic_\w+)\s*\(\s*([A-Za-z_]\w*)")


def stored_args(src_code):
    """The argument names `src_code` actually WRITES.

    Read off the source, not Inductor's tables: two of SD1.5's twelve
    in_out_ptr0 kernels never store to it, and the verify then reports a ghost.
    """
    return {m.group(2) for m in _STORE_RE.finditer(src_code)}


def demote_unwritten_inout(meta, src_code):
    """Turn an `inout` the kernel never stores to back into a plain `in`.

    In place, and returns it. Only `inout` is touched: an `out` with no store
    is a kernel that produces nothing, which is a different fact.
    """
    stores = stored_args(src_code)
    for a in meta.get("args", ()):
        if a.get("role") == "inout" and a.get("name") not in stores:
            a["role"] = "in"
    return meta


ROLES_BY_KERNEL = {}


def record_roles(kernel_name, meta):
    ROLES_BY_KERNEL[kernel_name] = [a["role"] for a in meta.get("args", ())]


def writes_arg(kernel_name, position):
    """Does `kernel_name` write the tensor argument at `position`?

    True when nothing is recorded, so this narrows only where there is a
    measurement to narrow it with.
    """
    roles = ROLES_BY_KERNEL.get(kernel_name)
    if roles is None or position >= len(roles):
        return True
    return roles[position] in ("out", "inout")


def collect_meta(kernel, kernel_name):
    """Everything the compile step needs, as plain repr-able data.

    Must run while `V.graph` is live, i.e. inside define_kernel: by the time
    the compile callable fires it is gone.
    """
    triton_meta = dict(getattr(kernel, "triton_meta", None) or {})
    signature = dict(triton_meta.get("signature") or {})
    constants = dict(triton_meta.get("constants") or {})

    roles = _roles(kernel)
    arg_defs, _call_args, _precompile, arg_types = kernel.args.python_argdefs()

    # A COMBO KERNEL HAS NO triton_meta ON THE OBJECT -- see
    # `_signature_from_argdefs`. Fill only what is missing, so a kernel that
    # does carry one keeps every token exactly as Inductor wrote it.
    if not signature:
        signature = _signature_from_argdefs(arg_defs, arg_types)

    args = []
    for a in arg_defs:
        name = getattr(a, "name", str(a))
        role, buf = roles.get(name, (None, None))
        if role is None:
            continue
        args.append({
            "name": name,
            "role": role,
            "buffer": buf,
            "dtype": _DTYPE.get(signature.get(name, ""), None),
            "numel": _buffer_numel(buf) if buf else None,
        })

    numels = {}
    for prefix, val in (getattr(kernel, "numels", None) or {}).items():
        try:
            numels[f"{prefix}numel"] = int(V.graph.sizevars.size_hint(val))
        except Exception:
            numels[f"{prefix}numel"] = None

    return {
        "kernel_name": kernel_name,
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": launch.fixed_config_for(kernel, numels, args),
        "template_grid": launch.template_grid(kernel) or launch.combo_grid(kernel),
    }


def _signature_from_argdefs(arg_defs, arg_types):
    """Triton signature tokens read off the ARG TYPES, for a kernel with no
    `triton_meta`.

    A ComboKernel builds its triton_meta as a local inside `codegen_kernel` and
    writes it into the emitted source; it never lands on the object, so
    `collect_meta`'s usual read comes back empty and every argument loses its
    dtype. The types are still there -- `python_argdefs` returns one per arg,
    from `V.graph.get_dtype` -- so the token is derivable, and derived here
    rather than parsed back out of the generated text.
    """
    import torch

    table = _dtype_tokens()
    tokens = {}
    for a, t in zip(arg_defs, arg_types):
        if not isinstance(t, torch.dtype):
            continue
        token = table.get(t)
        if token:
            tokens[getattr(a, "name", str(a))] = token
    return tokens


#: torch dtype -> the pointer token triton's signature uses for it. Filled on
#: first use, because this module is imported before torch is guaranteed to be.
_TORCH_DTYPE_TOKEN = None


def _dtype_tokens():
    global _TORCH_DTYPE_TOKEN
    if _TORCH_DTYPE_TOKEN is None:
        import torch
        _TORCH_DTYPE_TOKEN = {
            torch.float64: "*fp64", torch.float32: "*fp32",
            torch.float16: "*fp16", torch.bfloat16: "*bf16",
            torch.int64: "*i64", torch.int32: "*i32", torch.int16: "*i16",
            torch.int8: "*i8", torch.uint8: "*u8", torch.bool: "*i1",
            torch.float8_e5m2: "*fp8e5", torch.float8_e4m3fn: "*fp8e4nv",
        }
    return _TORCH_DTYPE_TOKEN


def scalar_args(meta):
    """User scalar parameters, in kernel order, as [(name, c_type, value)].

    the linalg lowering keeps these ahead of its own six grid/pid arguments, so the
    wrapper must pass them or every later argument lands one slot early.
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


SPEC_TEMPLATE = '''\
"""Generated by PyTorchSimCodegen/kernel_spec.py -- do not edit.

Inductor kernel {kernel_name!r}, rewritten for the compiler pipeline: the
triton_heuristics autotuner decorator is stripped and its block sizes are pinned
as constexprs, so the launch shape is static. See kernel_spec.py for why.
"""
import importlib.util
import os
import sys

sys.path.insert(0, {compiler_dir!r})
from {spec_module} import KernelSpec, Arg  # noqa: E402

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
    # checked at the graph level by the test that ran torch.compile. the compiler's
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
    parallel_axes={parallel_axes!r},
    reference=reference,
    make_inputs=make_inputs,
    extra={{"scalar_args": {scalar_decls!r},
           "scalar_values": {scalar_values!r}}},
)
'''


def write_spec_file(src_code, meta, path, compiler_dir):
    """Write a compiler kernel file for this Inductor kernel. Returns `path`.

    A block Inductor fixed in the kernel BODY rather than taking as a parameter
    is skipped: passing it would not match the signature.
    """
    missing = [a["name"] for a in meta["args"] if not a["dtype"] or not a["numel"]]
    if missing:
        raise SpecIncomplete(
            f"{meta['kernel_name']}: no dtype/numel for {missing} -- "
            f"collect_meta could not resolve them from V.graph")

    signature = dict(meta["signature"])
    constexprs = dict(meta["constants"])
    for k, v in (meta.get("fixed_config") or {}).items():
        if k not in signature:
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
        f.write(source_rewrite.clamp_instead_of_wrap(
            source_rewrite.strip_for_compiler(src_code), meta["kernel_name"]))

    scalars = scalar_args(meta)
    text = SPEC_TEMPLATE.format(
        kernel_name=meta["kernel_name"],
        compiler_dir=compiler_dir,
        spec_module=f"{compiler_bridge.COMPILER_PKG}.contract.kernel_spec",
        triton_module=triton_module,
        signature=signature,
        constexprs=constexprs,
        args_body=args_body,
        make_inputs_body=make_inputs_body,
        grid=launch.grid_xyz(meta),
        parallel_axes=tuple(launch.launch_axes(meta)),
        scalar_decls=[(n, c) for n, c, _ in scalars],
        scalar_values={n: v for n, _, v in scalars},
    )
    with open(path, "w") as f:
        f.write(text)
    return path
