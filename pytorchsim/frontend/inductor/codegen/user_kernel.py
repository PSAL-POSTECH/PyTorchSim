"""A Triton kernel a model registers itself (torch.library.triton_op, wrap_triton) on npu: its spec
meta per launch and its torchsim_compile call, its source the kernel's own and its jit helpers'."""

from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V

from pytorchsim import config
from pytorchsim.frontend.inductor.codegen import kernel_metadata

_logger = config.setup_logger()


def user_kernel_compile(kernel, meta):
    """The torchsim_compile call for a user-defined Triton kernel, with no source text read back:
    the source is the kernel's own and its jit helpers' (Inductor's transitive closure), written
    as it is (meta["user_defined"] keeps the Inductor-source rewriters off it)."""
    from torch._inductor.codegen.wrapper import (
        user_defined_triton_kernel_transitive_closure_source_code)

    src = IndentedBuffer()
    src.splice("""
        import triton
        import triton.language as tl
    """, strip=True)
    src.writeline("@triton.jit")
    src.splice(user_defined_triton_kernel_transitive_closure_source_code(kernel), strip=True)
    return kernel_metadata.compile_call(src.getvalue(), meta, kernel.__name__)


def user_kernel_meta(kernel, kwargs, triton_meta, grid):
    """The spec meta of one launch of a user-defined Triton kernel: the signature and constants
    Inductor's triton_meta, each pointer's dtype and storage span its IR argument, its role
    written (inout) where torch's mutation analysis says the launch writes it, read (in)
    elsewhere, every pointer inout when that analysis is not at hand."""
    from pytorchsim.contract import storage_span
    from pytorchsim.backend.codecache import user_kernel_id

    hint = V.graph.sizevars.optimization_hint
    signature = {str(k): str(v) for k, v in triton_meta["signature"].items()}
    writes = _user_kernel_writes(kwargs)
    args, call_params = [], []
    for arg, token in signature.items():
        if token == "constexpr":
            continue
        call_params.append(arg)
        if not token.startswith("*"):
            continue
        node = kwargs[arg]
        layout = node.get_layout()
        args.append({
            "name": arg,
            "role": "inout" if writes is None or arg in writes else "in",
            "buffer": node.get_name(),
            "dtype": kernel_metadata._DTYPE[token],
            "numel": storage_span([int(hint(s)) for s in layout.size],
                                  [int(hint(s)) for s in layout.stride]),
        })
    return {
        "kernel_name": kernel.__name__, "signature": signature,
        "constants": dict(triton_meta.get("constants") or {}),
        "args": args, "call_params": call_params, "numels": {},
        "inside_reduction": False, "fixed_config": {},
        "template_grid": ([int(hint(g)) for g in grid] + [1, 1, 1])[:3],
        "user_defined": True, "user_kernel": user_kernel_id(kernel),
    }


def _user_kernel_writes(kwargs):
    """The formal arguments the launch writes, as torch's analysis of the kernel's TTIR found them
    for its IR node (UserDefinedTritonKernel.arg_accesses, every tensor when the analysis failed),
    or None, with a warning, when the node being generated is not the launch's."""
    from torch._inductor import ir
    node = getattr(getattr(V.graph.scheduler, "current_node", None), "node", None)
    if not isinstance(node, ir.UserDefinedTritonKernel) or node.kwargs is not kwargs:
        _logger.warning("a user-defined Triton kernel's IR node is not the scheduler's current "
                        "node; every pointer argument is taken as written (inout)")
        return None
    return {dep.name for dep in node.arg_accesses.read_writes.writes}
