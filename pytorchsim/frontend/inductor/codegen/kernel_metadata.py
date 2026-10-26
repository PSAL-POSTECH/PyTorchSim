"""What the compile step needs from an Inductor kernel, read while `V.graph` is live: collect_meta
gathers the arguments with their roles, dtypes and storage spans, the numels, the pinned blocks and the grid;
define_kernel adds what it holds (graph_prefetch), none until then."""

from torch._inductor.codegen.common import InplacedBuffer, is_buffer_removed
from torch._inductor.virtualized import V

from pytorchsim.frontend import tiling
from pytorchsim.frontend.inductor.autotune.generated_kernel_blocks import blocks_for
from pytorchsim import config
from pytorchsim.contract import SpecIncomplete, storage_span

_logger = config.setup_logger()

#: Triton signature token -> the dtype's torch name: the one table of them on this side, holding the
#: dtypes the compiler's contract (kernel_object.ITEMSIZE) carries.
TOKEN_DTYPE = {
    "*fp64": "float64", "*fp32": "float32", "*fp16": "float16",
    "*bf16": "bfloat16",
    "*i64": "int64", "*i32": "int32", "*i16": "int16", "*i8": "int8",
    "*u64": "uint64", "*u32": "uint32", "*u16": "uint16", "*u8": "uint8",
    "*i1": "bool",
    "fp32": "float32", "i32": "int32", "i64": "int64",
}


def _buffer_numel(name):
    """How many elements of STORAGE an Inductor buffer spans, or None (logged) when its layout is not
    static. NOT the product of its shape: the kernel addresses with the strides, and a layout with
    gaps reaches further (ViT's buf18, 465792 against 465708)."""
    buf = V.graph.get_buffer(name)
    if buf is None:
        return None
    lay = buf.get_layout()
    hint = V.graph.sizevars.optimization_hint
    try:
        size = [int(hint(s)) for s in lay.size]
        stride = [int(hint(s)) for s in lay.stride]
        offset = int(hint(getattr(lay, "offset", 0)))
    except (AttributeError, TypeError, ValueError) as e:
        _logger.info("[torchsim-compile] %s has no static layout (%s): its storage span is unknown",
                     name, e)
        return None
    if any(s <= 0 for s in size):
        return 0
    return storage_span(size, stride, offset)


def _buffer_sizes(name):
    """The logical shape of an Inductor buffer, or None when it is not static: the dimensions a power
    model counting an operator's MACs needs, which the storage span loses."""
    buf = V.graph.get_buffer(name)
    if buf is None:
        return None
    try:
        return [int(V.graph.sizevars.optimization_hint(s)) for s in buf.get_layout().size]
    except (AttributeError, TypeError, ValueError):
        return None


def aten_ops(node_schedule):
    """The ATen ops the kernel's nodes came from, e.g. ["aten.mm.default"]: what Inductor's
    `Original ATen` comment names, as data."""
    return sorted({str(getattr(o, "target", o)) for n in node_schedule
                   for o in getattr(getattr(n, "node", None), "origins", ()) or ()
                   if getattr(o, "op", None) == "call_function"})


def _writes(kernel):
    """names -> whether `kernel` stores to any of them: the stores Inductor renders."""
    stored = set()
    with V.set_kernel_handler(kernel):
        stored |= {n for n in getattr(kernel, "store_buffer_names", ()) if not is_buffer_removed(n)}
    scheduler = getattr(V.graph, "scheduler", None)
    real = scheduler.mutation_real_name if scheduler else {}
    stored |= {real.get(n, n) for n in stored}
    return lambda names: any(n in stored or real.get(n, n) in stored for n in names)


def _roles(kernel):
    """arg name -> ('in'|'out'|'inout', buffer), from the kernel's tables: inplace_buffers and the
    mutations of buffers earlier kernels produced are inout, unless the kernel never stores to them."""
    writes = _writes(kernel)
    out = {}
    for buf, arg in getattr(kernel.args, "input_buffers", {}).items():
        out[arg] = ("in", buf)
    for buf, arg in getattr(kernel.args, "output_buffers", {}).items():
        out[arg] = ("out", buf)
    for buf, arg in getattr(kernel.args, "inplace_buffers", {}).items():
        names = arg.other_names if isinstance(arg, InplacedBuffer) else (buf,)
        name = getattr(arg, "inner_name", arg)
        out[name] = ("inout" if writes(names) else "in", buf)
    outputs = getattr(kernel.args, "output_buffers", {})
    for buf in getattr(kernel, "mutations", ()):
        arg = outputs.get(buf)
        name = getattr(arg, "inner_name", arg)
        if isinstance(name, str) and name in out:
            out[name] = ("inout" if writes((buf,)) else "in", buf)
    return out


ROLES_BY_KERNEL = {}


def record_roles(kernel_name, meta):
    ROLES_BY_KERNEL[kernel_name] = [a["role"] for a in meta.get("args", ())]


def writes_arg(kernel_name, position):
    """Whether `kernel_name` writes the tensor argument at `position`; True when nothing is recorded."""
    roles = ROLES_BY_KERNEL.get(kernel_name)
    if roles is None or position >= len(roles):
        return True
    return roles[position] in ("out", "inout")


def _held_blocks(kernel):
    """The block sizes Inductor generated this kernel with, or None for a kernel that has none (a template)."""
    held = getattr(getattr(kernel, "fixed_config", None), "config", None)
    return {k: v for k, v in held.items() if k.endswith("BLOCK")} if held else None


def _blocks_without_pin(kernel, numels, args):
    """Blocks for a kernel holding none: a TritonNPUKernel's from blocks_for, the answer pinning
    gives; a template's from its numels and args alone."""
    from pytorchsim.frontend.inductor.codegen.scheduling import TritonNPUKernel
    if isinstance(kernel, TritonNPUKernel) and all(isinstance(v, int) for v in numels.values()):
        return blocks_for(numels, kernel.features)
    return tiling.fixed_config_for(kernel, numels, args)


def compile_call(src_code, meta, kernel_name):
    """The wrapper's line compiling one kernel, torchsim_compile(source, meta=..., kernel_name=...):
    the source as a Python literal, so a quote or backslash in it reaches the compiler as written."""
    return f"torchsim_compile({src_code!r},\n    meta={meta!r},\n    kernel_name={kernel_name!r})"


def collect_meta(kernel, kernel_name):
    """Everything the compile step needs, as plain repr-able data; runs inside define_kernel, while
    `V.graph` is live."""
    triton_meta = dict(getattr(kernel, "triton_meta", None) or {})
    signature = dict(triton_meta.get("signature") or {})
    constants = dict(triton_meta.get("constants") or {})

    roles = _roles(kernel)
    arg_defs, _call_args, _precompile, _arg_types = kernel.args.python_argdefs()

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
            "dtype": TOKEN_DTYPE.get(signature.get(name, ""), None),
            "numel": _buffer_numel(buf) if buf else None,
            "sizes": _buffer_sizes(buf) if buf else None,
        })

    numels = {}
    for prefix, val in (getattr(kernel, "numels", None) or {}).items():
        try:
            numels[f"{prefix}numel"] = int(V.graph.sizevars.optimization_hint(val))
        except (TypeError, ValueError):
            numels[f"{prefix}numel"] = None

    return {
        "kernel_name": kernel_name,
        "signature": {str(k): str(v) for k, v in signature.items()},
        "constants": {str(k): v for k, v in constants.items()},
        "args": args,
        "call_params": call_params,
        "numels": numels,
        "inside_reduction": bool(getattr(kernel, "inside_reduction", False)),
        "fixed_config": _held_blocks(kernel) or _blocks_without_pin(kernel, numels, args),
        "template_grid": template_grid(kernel),
        "graph_prefetch": [],
    }


def benchmark_request_meta(req):
    """(source, meta) of a template choice as Inductor rendered it to benchmark: the module's source,
    the signature off its loaded kernel, the roles and spans off the request's TensorMeta; the same
    fields collect_meta gives the kernel the choice becomes."""
    from torch._inductor.codecache import PyCodeCache

    mod = PyCodeCache.load_by_key_path(req.module_cache_key, req.module_path)
    triton_meta = getattr(mod, req.kernel_name).triton_meta
    with open(req.module_path) as f:
        src = f.read()
    signature = {str(k): str(v) for k, v in triton_meta["signature"].items()}
    tensors = list(req.input_tensor_meta) + [req.output_tensor_meta]
    roles = ["in"] * len(req.input_tensor_meta) + ["out"]
    ptrs = [n for n, t in signature.items() if t.startswith("*")]
    args = [{"name": n, "role": r, "buffer": t.name, "dtype": TOKEN_DTYPE.get(signature[n]),
             "numel": storage_span(list(t.sizes), list(t.strides), t.offset),
             "sizes": [int(s) for s in t.sizes]}
            for n, r, t in zip(ptrs, roles, tensors)]
    xnumel = 1
    for s in req.output_tensor_meta.sizes:
        xnumel *= s
    numels = {"xnumel": xnumel, "r0_numel": 1}
    return src, {
        "kernel_name": req.kernel_name,
        "signature": signature,
        "constants": {str(k): v for k, v in (triton_meta.get("constants") or {}).items()},
        "args": args,
        "call_params": list(signature),
        "numels": numels,
        "inside_reduction": False,
        "fixed_config": tiling.fixed_config_for(None, numels, args),
        "template_grid": [int(g) for g in req.extra_args],
        "graph_prefetch": [],
    }


def template_grid(kernel):
    """A template kernel's launch grid as [gridX, gridY, gridZ], else None: mm and conv tile with their
    own BLOCK_M/N, so the grid is theirs to state; refuses rather than falling back."""
    grid_fn = getattr(kernel, "grid_fn", None)
    sizes = getattr(kernel, "call_sizes", None)
    if grid_fn is None or sizes is None:
        return None
    name = getattr(kernel, "kernel_name", kernel)
    try:
        vals = [int(V.graph.sizevars.optimization_hint(s)) for s in sizes]
        grid = grid_fn(*vals, dict(getattr(kernel, "meta", None) or {}))
        extents = [int(g) for g in grid]
    except Exception as e:
        raise SpecIncomplete(
            f"{name}: template grid_fn raised {type(e).__name__}: {e}") from e
    if len(extents) != 3 or any(g < 1 for g in extents):
        raise SpecIncomplete(
            f"{name}: template grid_fn returned {grid!r}; this route needs three "
            f"static positive extents. Refusing rather than falling back to the "
            f"numels, which describe the output tensor and not the grid.")
    return extents
