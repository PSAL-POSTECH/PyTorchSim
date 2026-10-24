"""NPUChoices, Inductor's choices class (torch._inductor.choices, config.inductor_choices_class):
the questions Inductor asks while lowering and scheduling, answered for npu by autotune/ (template
kernels' configs and recorded picks, generated kernels' reductions) and by Inductor elsewhere."""

import functools


def install():
    """NPUChoices as Inductor's choices, unless something else already is."""
    from torch._inductor import config as inductor_config

    if inductor_config.inductor_choices_class is None:
        inductor_config.inductor_choices_class = npu_choices_class()


_NPU_CHOICES = None


def npu_choices_class():
    """NPUChoices, built once: Inductor's lookup-table choices whose npu answers come from autotune/
    (template kernels' configs and recorded picks, generated kernels' reductions); every other
    device keeps Inductor's."""
    global _NPU_CHOICES
    if _NPU_CHOICES is not None:
        return _NPU_CHOICES
    from torch._inductor.choices import InductorChoices
    from torch._inductor.lookup_table import LookupTableChoices

    from pytorchsim.backend import template_timing
    from pytorchsim.frontend.inductor import is_lowering_npu
    from pytorchsim.frontend.inductor.autotune import (generated_kernel_blocks,
                                                       template_kernel_configs)

    class NPUChoices(LookupTableChoices):
        def uuid(self):
            """This class's part of Inductor's cache key: the machine and the picks it hands out."""
            return f"pytorchsim-npu-choices-4-{template_timing.graph_digest()}"

        def get_config_heuristics(self, device_type="cuda"):
            if device_type == "npu":
                return template_kernel_configs.conv_config_heuristic()
            return super().get_config_heuristics(device_type)

        def _get_lookup_table(self):
            """This machine's recorded picks; Inductor's own looks only on CUDA."""
            return template_timing.lookup_table()

        @staticmethod
        def _get_device_key(device):
            if getattr(device, "type", None) == "npu":
                return template_timing.device_key()
            return LookupTableChoices._get_device_key(device)

        def _create_lookup_choices(self, *args, **kwargs):
            """Inductor's choices from the recorded configs, each marked as recorded."""
            ktcs = super()._create_lookup_choices(*args, **kwargs)
            for ktc in ktcs:
                ktc.annotations[template_timing.SOURCE] = template_timing.RECORDED
            return ktcs

        def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
            """This machine's blocks pinned into every npu kernel Inductor generates (Inductor's
            hook for a fixed config): the one it emits and the one its fusion benchmark times are
            made the same way, so they are the same kernel."""
            kwargs = super().triton_kernel_kwargs(kernel_cls, features, groups, kernel_kwargs)
            if is_lowering_npu() and kwargs.get("fixed_config") is None:
                kwargs["fixed_config"] = generated_kernel_blocks.pinned_blocks(groups[0], features)
            return kwargs

        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            return generated_kernel_blocks.use_persistent_reduction(
                features,
                InductorChoices.should_use_persistent_reduction(features, cooperative_reduction))

        def reduction_split_factor(self, device, reduction_numel_hint, numel_hint, inner_reduction):
            upstream = functools.partial(super().reduction_split_factor, device,
                                         reduction_numel_hint, numel_hint, inner_reduction)
            if getattr(device, "type", device) != "npu":
                return upstream()
            return generated_kernel_blocks.reduction_split(reduction_numel_hint, numel_hint,
                                                           upstream)

    NPUChoices.__module__ = __name__
    NPUChoices.__qualname__ = "NPUChoices"
    globals()["NPUChoices"] = NPUChoices
    _NPU_CHOICES = NPUChoices
    return NPUChoices
