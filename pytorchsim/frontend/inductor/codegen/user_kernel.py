"""A Triton kernel a model registers itself (torch.library.triton_op, wrap_triton) on npu: the spec of each
launch, which Inductor's own async_compile.triton call carries to the Triton backend, and the source and meta a
kernel file of it is written from."""

import dis
import types

from torch._inductor.virtualized import V

from pytorchsim import config

_logger = config.setup_logger()


def user_kernel_spec(kernel, kwargs, grids):
    """The spec JSON of one launch of a user-defined Triton kernel: each tensor argument (kernel parameter
    order) by role, where torch's mutation analysis says the launch writes it (every pointer without that
    analysis), and storage span; the grid of the config with the most programs per axis."""
    import json
    from torch._inductor import ir
    from pytorchsim.backend import compiler_input
    from pytorchsim.contract import storage_span

    hint = V.graph.sizevars.optimization_hint
    writes = _user_kernel_writes(kwargs)
    tensors = []
    for p in kernel.params:
        arg = kwargs.get(p.name)
        if not isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
            continue
        layout = arg.get_layout()
        tensors.append(("inout" if writes is None or p.name in writes else "in",
                        storage_span([int(hint(s)) for s in layout.size],
                                     [int(hint(s)) for s in layout.stride])))
    extents = [max(([int(hint(g)) for g in grid] + [1, 1, 1])[i] for grid in grids) for i in range(3)]
    return json.dumps(compiler_input.spec_dict(tensors, extents, axes=("z", "y", "x")))


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


def user_kernel_source(kernel):
    """The module text of a user-defined Triton kernel as a kernel file loads it: the kernel's own source and its
    jit helpers' (Inductor's transitive closure), with the Triton modules they read as globals imported."""
    from torch._inductor.codegen.common import IndentedBuffer
    from torch._inductor.codegen.wrapper import user_defined_triton_kernel_transitive_closure_source_code

    src = IndentedBuffer()
    src.splice("""
        import triton
        import triton.language as tl
    """, strip=True)
    for line in _extra_imports(kernel):
        src.writeline(line)
    src.writeline("@triton.jit")
    src.splice(user_defined_triton_kernel_transitive_closure_source_code(kernel), strip=True)
    return src.getvalue()


def _extra_imports(kernel):
    """An import of each Triton module other than triton and tl that the kernel or a jit helper it calls
    reads as a global, under that name (`pytorchsim` for triton.language.extra.pytorchsim), found as
    triton finds the kernel's unqualified loads; Inductor's closure imports a module global by __module__."""
    from triton.runtime.jit import JITFunction

    lines, seen, todo = [], set(), [kernel]
    while todo:
        fn = getattr(todo.pop(), "fn", None)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for name in {i.argval for i in dis.Bytecode(fn) if i.opname == "LOAD_GLOBAL"}:
            value = fn.__globals__.get(name)
            if (isinstance(value, types.ModuleType) and name not in ("triton", "tl")
                    and value.__name__.split(".")[0] == "triton"):
                line = f"import {value.__name__} as {name}"
                if line not in lines:
                    lines.append(line)
            elif isinstance(value, JITFunction):
                todo.append(value)
    return sorted(lines)


def user_kernel_meta(kernel, kwargs, triton_meta, grid):
    """The spec meta of one launch of a user-defined Triton kernel: signature and constants from
    Inductor's triton_meta, each pointer's dtype and span from its IR argument, and its role inout
    where torch's mutation analysis says the launch writes it (every pointer, without that analysis)."""
    from pytorchsim.contract import storage_span
    from pytorchsim.backend.launch_hooks import user_kernel_id
    from pytorchsim.frontend.inductor.autotune import graph_prefetch, on_chip
    from pytorchsim.frontend.inductor.codegen import kernel_metadata

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
            "dtype": kernel_metadata.TOKEN_DTYPE[token],
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
        "graph_prefetch": graph_prefetch.statement(kernel, args),
        "on_chip": on_chip.statement(args),
    }
