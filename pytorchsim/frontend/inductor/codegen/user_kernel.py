"""A Triton kernel a model registers itself (torch.library.triton_op, wrap_triton) on npu: the spec of each
launch, which Inductor's own async_compile.triton call carries to the Triton backend."""

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
