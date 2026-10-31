"""scatter_chain: a chain of npu view scatters (a loop of x[..., i, :i] = v) written in place, Inductor's
reinplace patched (it has no hook); here so the pass's uuid, which hashes this package, covers it."""

import dataclasses

import torch


# scatter_chain: a view scatter onto a scatter's result, reinplaced on the npu

def install_scatter_chain():
    """A view scatter onto another scatter's result reinplaces on the npu when reinplace may: upstream
    writes it functionally unless both are realized, a full copy of x per write."""
    from torch._inductor.fx_passes import reinplace

    scatter = reinplace._generalized_scatter
    op = reinplace.inplaceable_ops[scatter]
    if getattr(op.extra_check, "npu_chain", False):
        return
    upstream = op.extra_check

    def check(node):
        base = reinplace._get_view_base(node.args[0])
        val = node.meta.get("val", None)
        on_npu = isinstance(val, torch.Tensor) and val.device.type == "npu"
        chained = base.target in (scatter, reinplace._inplace_generalized_scatter)
        return (on_npu and chained) or upstream(node)

    check.npu_chain = True
    check.upstream = upstream
    reinplace.inplaceable_ops[scatter] = dataclasses.replace(op, extra_check=check)
