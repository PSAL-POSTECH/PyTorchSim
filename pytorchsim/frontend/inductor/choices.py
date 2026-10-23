"""The choices npu makes differently from Inductor's, through Inductor's own hooks: NPUChoices
(config.inductor_choices_class) for conv tiles, persistent reductions, splits and recorded template
picks, and register_template_heuristic for the mm/bmm tiles.
"""

import threading

from pytorchsim import config
from pytorchsim.frontend import tiling
from pytorchsim.frontend.inductor.blocks import _elem_args, _kept_axis_contiguous
from pytorchsim.frontend.inductor.registration import lowering_npu

logger = config.setup_logger()

conv_groups = threading.local()

_NPU_CHOICES = None


def install():
    """Make NPUChoices Inductor's choices, unless something else already is, and register the
    mm/bmm/addmm/baddbmm heuristics."""
    from torch._inductor import config as inductor_config

    if inductor_config.inductor_choices_class is None:
        inductor_config.inductor_choices_class = _npu_choices_class()
    _register_template_heuristics()


def _npu_choices_class():
    """NPUChoices, built once: Inductor's lookup-table choices, with this machine's conv tiles,
    reduction policy and recorded template picks."""
    global _NPU_CHOICES
    if _NPU_CHOICES is not None:
        return _NPU_CHOICES
    from torch._inductor.choices import InductorChoices
    from torch._inductor.lookup_table import LookupTableChoices

    from pytorchsim.backend import select

    class NPUChoices(LookupTableChoices):
        def uuid(self):
            """This class's part of Inductor's cache key: the machine and the picks it hands out."""
            return f"pytorchsim-npu-choices-4-{select.graph_digest()}"

        def get_config_heuristics(self, device_type="cuda"):
            """npu's conv tiles (_npu_config_heuristic); every other device keeps Inductor's."""
            if device_type == "npu":
                return _npu_config_heuristic()
            return super().get_config_heuristics(device_type)

        def _get_lookup_table(self):
            """This machine's recorded picks; Inductor's own looks only on CUDA."""
            return select.lookup_table()

        @staticmethod
        def _get_device_key(device):
            if getattr(device, "type", None) == "npu":
                return select.device_key()
            return LookupTableChoices._get_device_key(device)

        def _create_lookup_choices(self, *args, **kwargs):
            """Inductor's choices from the recorded configs, each marked as recorded."""
            ktcs = super()._create_lookup_choices(*args, **kwargs)
            for ktc in ktcs:
                ktc.annotations[select.SOURCE] = select.RECORDED
            return ktcs

        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            """Persistent only when the whole reduction fits this machine's reduction block."""
            base = InductorChoices.should_use_persistent_reduction(
                features, cooperative_reduction)
            if not base or not lowering_npu():
                return base
            try:
                extent = int(features.reduction_numel)
            except (TypeError, ValueError):
                return base
            if extent < 1:
                return base
            persistent = 1 << (extent - 1).bit_length()
            try:
                numels = {"xnumel": int(features.numel), "r0_numel": extent}
                shim = type("K", (), {"inside_reduction": True, "numels": numels})()
                pinned = tiling.fixed_config_for(shim, numels, _elem_args(features),
                                                 contiguous=_kept_axis_contiguous(features))
                return persistent <= int(pinned["R0_BLOCK"])
            except Exception:  # noqa: BLE001
                return persistent <= tiling.reduction_block_for(extent)

        def reduction_split_factor(self, device, reduction_numel_hint, numel_hint,
                                   inner_reduction):
            """Split a reduction only when a core would otherwise sit idle, and at most cores ways.

            One core, or a parallel axis that already fills the cores, means no split.
            """
            if getattr(device, "type", device) != "npu":
                return super().reduction_split_factor(
                    device, reduction_numel_hint, numel_hint, inner_reduction)
            cores = _num_cores()
            if cores > 1:
                want = super().reduction_split_factor(
                    device, reduction_numel_hint, numel_hint, inner_reduction)
                try:
                    if int(numel_hint) >= cores:
                        return 1
                except (TypeError, ValueError):
                    return 1
                split = min(int(want), cores)
                if split > 1:
                    logger.info(
                        "[torchsim-compile] splitting a reduction of %s elements %s ways "
                        "over %s cores (upstream wanted %s)",
                        reduction_numel_hint, split, cores, want)
                    return split
                return 1
            return 1

    NPUChoices.__module__ = __name__
    NPUChoices.__qualname__ = "NPUChoices"
    globals()["NPUChoices"] = NPUChoices
    _NPU_CHOICES = NPUChoices
    return NPUChoices


def _npu_config_heuristic():
    """npu's conv tile table: BLOCK_N is the lane count, per group (groups_now)."""
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
            base = super().get_conv_configs()

            def per_group(m, n, k, **kwargs):
                groups = groups_now()
                return base(m, max(1, n // groups) if groups > 1 else n, k,
                            **kwargs)

            return per_group

    return NPUConfigHeuristic()


def groups_now():
    """The groups of the convolution being lowered right now (1 outside one)."""
    return getattr(conv_groups, "value", 1) or 1


def _register_template_heuristics():
    """Register npu's mm/bmm/addmm/baddbmm tile heuristics: this machine's tiles, then the generic set."""
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.heuristics.template.registry import (
        register_template_heuristic)
    from torch._inductor.heuristics.template.triton import (
        BaseConfigHeuristic, MMTemplateConfigMixin)
    from torch._inductor.heuristics.template.triton_addmm import AddMMConfigMixin

    @register_template_heuristic(mm_template.uid, "npu")
    @register_template_heuristic(bmm_template.uid, "npu")
    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        def _get_config_generator(self):
            """Configs for one m, n, k: the first codegen_autotune_template_topk of gemm_tiles,
            or the generic set when none of this machine's tiles is a legal block."""
            generic = super()._get_config_generator()

            def configs(m, n, k, **kwargs):
                from torch._inductor.virtualized import V
                try:
                    mnk = [int(V.graph.sizevars.optimization_hint(s)) for s in (m, n, k)]
                except Exception:
                    yield from generic(m, n, k, **kwargs)
                    return
                mapped = tiling.gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                if not mapped:
                    logger.warning(
                        "[torchsim-compile] no mapped tile for %sx%sx%s is a legal "
                        "Triton block; falling back to the generic set", *mnk)
                    yield from generic(m, n, k, **kwargs)
                    return
                yield from self._finalize_mm_configs(
                    mapped[:config.codegen_autotune_template_topk])

            return configs

    @register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        pass


def _num_cores():
    """This machine's core count, `num_cores` of its TOGSim config (every machine file has it)."""
    return max(1, int(config.CONFIG_NUM_CORES))
