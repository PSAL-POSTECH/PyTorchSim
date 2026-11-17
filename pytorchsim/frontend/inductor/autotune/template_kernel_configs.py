"""The configs a template kernel (mm, bmm, addmm, baddbmm, conv) is offered on npu: this machine's
mm tiles, for an ungrouped conv those of one tap's GEMM. A conv's groups reach the heuristic
through the node being lowered and size its grid per group, a template's grid Inductor has no hook for."""

import torch

from pytorchsim import config
from pytorchsim.frontend import tiling
from pytorchsim.frontend.inductor.autotune import template_kernel_spec
from pytorchsim.frontend.inductor import is_lowering_npu

logger = config.setup_logger()

_CONVOLUTIONS = (torch.ops.aten.convolution, *(getattr(torch.ops.aten.convolution, o)
                                              for o in torch.ops.aten.convolution.overloads()))


def install():
    """Grid a grouped conv per group (Inductor's conv templates take no grid but the module's); the heuristics are
    registered by registration.py."""
    _size_grouped_conv_grid_per_group()


def mm_config_heuristics():
    """(mm/bmm, addmm/baddbmm) heuristic classes: this machine's tiles, then the generic set;
    registration.py registers them for npu."""
    from torch._inductor.heuristics.template.triton import (
        BaseConfigHeuristic, MMTemplateConfigMixin)
    from torch._inductor.heuristics.template.triton_addmm import AddMMConfigMixin

    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        bias = False

        def _offered_tiles(self, m, n, k, dtype_size):
            """The tiles one mm is offered, best first: the first codegen_autotune_template_topk of gemm_tiles
            where the strategy autotunes, else tiling.GEMM_CANDIDATES of them, each compiled until one fits."""
            from pytorchsim.backend import template_record
            limit = config.codegen_autotune_template_topk if template_record.autotunes() \
                else tiling.GEMM_CANDIDATES
            return tiling.gemm_tiles(m, n, k, dtype_size, bias=self.bias)[:limit]

        def get_extra_kwargs(self, kernel_inputs, op_name):
            """The mm kwargs and the spec its choices compile with (template_kernel_spec)."""
            from torch._inductor.virtualized import V
            kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
            mnk = [int(V.graph.sizevars.optimization_hint(s)) for s in kernel_inputs.mnk_symbolic()]
            tiles = [(t.block_m, t.block_n) for t in
                     self._offered_tiles(*mnk, kernel_inputs.dtype().itemsize)]
            if not tiles:
                return kwargs
            grids = [template_kernel_spec.mm_grid(kernel_inputs, t) for t in tiles]
            return {**kwargs, "triton_meta": template_kernel_spec.triton_meta_for(kernel_inputs, grids)}

        def _get_config_generator(self):
            """Configs for one m, n, k: the first codegen_autotune_template_topk of gemm_tiles (the
            first alone unless the strategy autotunes), or the generic set when none is a legal block."""
            generic = super()._get_config_generator()

            def configs(m, n, k, **kwargs):
                from torch._inductor.virtualized import V
                mnk = [int(V.graph.sizevars.optimization_hint(s)) for s in (m, n, k)]
                mapped = tiling.gemm_tiles(*mnk, kwargs.get("dtype_size", 4), bias=self.bias)
                if not mapped:
                    logger.warning(
                        "[torchsim-compile] no mapped tile for %sx%sx%s is a legal "
                        "Triton block; falling back to the generic set", *mnk)
                    yield from generic(m, n, k, **kwargs)
                    return
                yield from self._finalize_mm_configs(self._offered_tiles(*mnk, kwargs.get("dtype_size", 4)))

            return configs

    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        bias = True

    return NPUMMTemplateConfigHeuristic, NPUAddmmTemplateConfigHeuristic


def mm_resident_config_heuristic():
    """(mm, addmm) heuristic classes of the resident mm: its one config from the compiler's copy_node.plan
    (mm_resident.configs), the addmm form with the bias epilogue; registration.py registers them for npu."""
    from torch._inductor.heuristics.template.base import TemplateConfigHeuristics
    from torch._inductor.heuristics.template.triton_addmm import AddMMConfigMixin

    from pytorchsim.frontend.inductor.autotune import mm_resident

    class NPUResidentMMConfigHeuristic(TemplateConfigHeuristics):
        def _get_template_configs_impl(self, kernel_inputs, op_name):
            yield from mm_resident.configs(kernel_inputs)

        def get_extra_kwargs(self, kernel_inputs, op_name):
            """The spec the resident mm compiles with: one program over the whole output."""
            return {"triton_meta": template_kernel_spec.triton_meta_for(kernel_inputs, [(1, 1, 1)])}

    class NPUResidentAddmmConfigHeuristic(AddMMConfigMixin, NPUResidentMMConfigHeuristic):
        pass

    return NPUResidentMMConfigHeuristic, NPUResidentAddmmConfigHeuristic


#: The conv template's widest BLOCK_M: its X tile is a gather, whose index scratch grows with the rows.
CONV_MAX_BLOCK_M = 2048


def conv_config_heuristic():
    """npu's conv tiles: this machine's mm tiles (tiling.gemm_tiles) for one tap's GEMM, pixels x out channels
    x in channels, BLOCK_M at most CONV_MAX_BLOCK_M, the first codegen_autotune_template_topk of them where
    the strategy autotunes; the lane table (BLOCK_N per group) for a grouped conv or when no tile fits."""
    from torch._inductor.heuristics.template.triton import (
        BaseConfigHeuristic, ConvConfig)

    lanes = int(config.CONFIG_MACHINE.vector_lanes)

    class NPUConfigHeuristic(BaseConfigHeuristic):
        def __init__(self):
            super().__init__()
            self.conv_configs = [
                ConvConfig(64, lanes, 16, 1, 4),
                ConvConfig(32, lanes, 16, 1, 4),
                ConvConfig(64, lanes, 32, 1, 4),
            ]

        def get_conv_configs(self):
            def per_group(m, n, k, **kwargs):
                """Every tile a timed run ranks, the first alone otherwise: one candidate is picked by the
                order it was offered, which is no ranking (the benchmarker ranks the rest)."""
                from torch._inductor.virtualized import V
                from pytorchsim.backend import template_record
                groups = conv_groups_now()
                mnk = [int(V.graph.sizevars.optimization_hint(s)) for s in (m, n, k)]
                topk = config.codegen_autotune_template_topk if template_record.autotunes() else 1
                tiles = [] if groups > 1 else [
                    t for t in tiling.gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                    if t.block_m <= CONV_MAX_BLOCK_M][:topk]
                configs = [ConvConfig(t.block_m, t.block_n, t.block_k, 1, 4) for t in tiles]
                if groups == 1 and not configs:
                    logger.warning("[torchsim-compile] no mapped conv tile for %sx%sx%s; the lane table", *mnk)
                return self.preprocess_mm_configs(m, max(1, n // groups) if groups > 1 else n, k,
                                                  configs=configs or self.conv_configs, op_name="conv",
                                                  **kwargs)

            return per_group

    return NPUConfigHeuristic()


def conv_groups_now():
    """The groups of the aten.convolution being lowered right now (1 outside one): the lowering hands the conv
    heuristic its channels, not its groups, so they are read off the node V.graph is lowering."""
    from torch._inductor.virtualized import V

    node = getattr(V.graph, "current_node", None)
    if node is None or node.target not in _CONVOLUTIONS:
        return 1
    groups = node.kwargs.get("groups", node.args[8] if len(node.args) > 8 else 1)
    return groups if isinstance(groups, int) else 1


def _size_grouped_conv_grid_per_group():
    """Grid a grouped convolution over one group's channels, not all of them: the two conv templates' grid, an
    upstream grid off the npu (the module's own functions are left as they are)."""
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    upstream = {"conv2d_template": conv_kernel.conv2d_grid, "conv3d_template": conv_kernel.conv3d_grid}

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return upstream["conv2d_template"](n, c, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return upstream["conv3d_template"](n, c, d, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_template.grid = conv2d_grid
    conv_kernel.conv3d_template.grid = conv3d_grid
