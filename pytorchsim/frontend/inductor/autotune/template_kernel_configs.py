"""The configs a template kernel (mm, bmm, addmm, baddbmm, conv) is offered on npu: this machine's
mm tiles, for an ungrouped conv those of one tap's GEMM. A conv's groups reach the heuristic
through its lowering and size its grid per group, two patches Inductor has no hook for."""

import threading

import torch

from pytorchsim import config
from pytorchsim.frontend import tiling
from pytorchsim.frontend.inductor import is_lowering_npu

logger = config.setup_logger()

conv_groups = threading.local()


def install():
    """Apply the two conv patches (Inductor has no hook for either); the heuristics are registered
    by registration.py."""
    _record_conv_groups()
    _size_grouped_conv_grid_per_group()


def mm_config_heuristics():
    """(mm/bmm, addmm/baddbmm) heuristic classes: this machine's tiles, then the generic set;
    registration.py registers them for npu."""
    from torch._inductor.heuristics.template.triton import (
        BaseConfigHeuristic, MMTemplateConfigMixin)
    from torch._inductor.heuristics.template.triton_addmm import AddMMConfigMixin

    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        bias = False

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
                from pytorchsim.backend import template_record
                topk = config.codegen_autotune_template_topk if template_record.autotunes() else 1
                yield from self._finalize_mm_configs(mapped[:topk])

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

    class NPUResidentAddmmConfigHeuristic(AddMMConfigMixin, NPUResidentMMConfigHeuristic):
        pass

    return NPUResidentMMConfigHeuristic, NPUResidentAddmmConfigHeuristic


#: The conv template's widest BLOCK_M: its X tile is a gather, whose index scratch grows with the rows.
CONV_MAX_BLOCK_M = 2048


def conv_config_heuristic():
    """npu's conv tiles: the largest of this machine's mm tiles (tiling.gemm_tiles) for one tap's GEMM, pixels x
    out channels x in channels, BLOCK_M at most CONV_MAX_BLOCK_M; the lane table (BLOCK_N per group) for a
    grouped conv, untried at mm tiles, or when no tile fits."""
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
                from torch._inductor.virtualized import V
                groups = conv_groups_now()
                mnk = [int(V.graph.sizevars.optimization_hint(s)) for s in (m, n, k)]
                tiles = [] if groups > 1 else [
                    t for t in tiling.gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                    if t.block_m <= CONV_MAX_BLOCK_M][:1]
                configs = [ConvConfig(t.block_m, t.block_n, t.block_k, 1, 4) for t in tiles]
                if groups == 1 and not configs:
                    logger.warning("[torchsim-compile] no mapped conv tile for %sx%sx%s; the lane table", *mnk)
                return self.preprocess_mm_configs(m, max(1, n // groups) if groups > 1 else n, k,
                                                  configs=configs or self.conv_configs, op_name="conv",
                                                  **kwargs)

            return per_group

    return NPUConfigHeuristic()


def conv_groups_now():
    """The groups of the convolution being lowered right now (1 outside one)."""
    return getattr(conv_groups, "value", 1) or 1


def _record_conv_groups():
    """Wrap aten.convolution's lowering to record its groups, which narrow the conv heuristic's BLOCK_N
    to one group's channels (conv_groups_now)."""
    import functools
    import inspect

    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel

    sig = inspect.signature(conv_kernel.convolution)

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            groups = bound.arguments.get("groups", 1)
            prev = getattr(conv_groups, "value", 1)
            conv_groups.value = groups if isinstance(groups, int) else 1
            try:
                return inner(*args, **kwargs)
            finally:
                conv_groups.value = prev

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_grouped_conv_grid_per_group():
    """Grid a grouped convolution over one group's channels, not all of them."""
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    orig2d = conv_kernel.conv2d_grid
    orig3d = conv_kernel.conv3d_grid

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return orig2d(n, c, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return orig3d(n, c, d, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_grid = conv2d_grid
    conv_kernel.conv3d_grid = conv3d_grid
    for tmpl, fn in ((getattr(conv_kernel, "conv2d_template", None), conv2d_grid),
                     (getattr(conv_kernel, "conv3d_template", None), conv3d_grid)):
        if tmpl is not None:
            tmpl.grid = fn
