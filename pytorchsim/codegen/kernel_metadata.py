"""What the compile step needs from an Inductor kernel, read while `V.graph` is live.

`collect_meta` gathers it as plain data: the arguments with their roles, dtypes and storage spans,
the numels, the pinned blocks and the grid.
"""

import re

from torch._inductor.virtualized import V

from pytorchsim.codegen import block_sizing
from pytorchsim.contract import storage_span

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
        hint = V.graph.sizevars.optimization_hint
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
        return storage_span(size, stride,
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


def _held_blocks(kernel):
    """The block sizes Inductor generated this kernel with, or None for a kernel that has none (a template)."""
    held = getattr(getattr(kernel, "fixed_config", None), "config", None)
    return {k: v for k, v in held.items() if k.endswith("BLOCK")} if held else None


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

    call_params = [getattr(a, "name", str(a)) for a in arg_defs]
    args = []
    for a in arg_defs:
        name = getattr(a, "name", str(a))
        role, buf = roles.get(name, (None, None))
        if role is None or not signature.get(name, "*").startswith("*"):
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
            numels[f"{prefix}numel"] = int(V.graph.sizevars.optimization_hint(val))
        except Exception:
            numels[f"{prefix}numel"] = None

    return {
        "kernel_name": kernel_name,
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "call_params": call_params,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": _held_blocks(kernel) or block_sizing.fixed_config_for(kernel, numels, args),
        "template_grid": block_sizing.template_grid(kernel) or block_sizing.combo_grid(kernel),
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
