"""Everything npu registers with Inductor, one section per registry it fills, in install()'s order:

    Dynamo's device interfaces    NPUDeviceInterface: what Inductor asks of the device
    GPU_TYPES                     npu as a Triton GPU (a patch: it has no hook)
    Inductor's device codegens    scheduling, wrapper, post-grad pass
    device op overrides           NPUDeviceOpOverrides: the device code in a wrapper
    Inductor's choices class      NPUChoices, answered by autotune/
    template heuristics           the mm/bmm configs and the resident mm's, autotune/'s
"""

import functools
from textwrap import dedent

import torch
from torch._dynamo.device_interface import DeviceInterface
from torch._inductor.codegen.common import DeviceOpOverrides

from pytorchsim import config


def install():
    """Register npu in every registry: the device interface first, since Inductor's Triton check
    (has_triton) asks it."""
    _register_device_interface()
    _register_triton_gpu()
    _register_device_codegen()
    _register_device_op_overrides()
    _register_choices()
    _register_template_heuristics()


# Dynamo's device interfaces (torch._dynamo.device_interface)

def _register_device_interface():
    """Register NPUDeviceInterface as npu's DeviceInterface."""
    from torch._dynamo.device_interface import register_interface_for_device

    register_interface_for_device("npu", NPUDeviceInterface)


class NPUDeviceProperties:
    """The properties Inductor reads; every other one it reads by getattr with a default."""

    def __init__(self):
        self.multi_processor_count = int(config.CONFIG_MACHINE.vector_lanes)


class NPUDeviceInterface(DeviceInterface):
    """npu's DeviceInterface: Dynamo's registry, read by Inductor (properties, compute capability,
    availability, Triton and bf16 support), answered for the simulator."""

    class Worker:
        @staticmethod
        def set_device(device: int):
            torch.npu.set_device(device)

        @staticmethod
        def current_device() -> int:
            return torch.npu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> NPUDeviceProperties:
            """One set for every npu device: the simulated machine's."""
            return NPUDeviceProperties()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int:
        """None of CUDA's; Triton's compile metadata carries it, and torchsim_compile ignores it."""
        return 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        """Kernels for npu are Triton, compiled by PyTorchSim-Triton-Backend."""
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True


# npu as a Triton GPU: GPU_TYPES, a patch Inductor has no hook for

def _register_triton_gpu():
    """The patch; has_triton and the backend hash are upstream's."""
    _add_gpu_type()


def _add_gpu_type():
    """Add npu to Inductor's GPU_TYPES, so Triton codegen is chosen for it."""
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


# Inductor's device codegens (register_backend_for_device)

def _register_device_codegen():
    """Register npu's scheduling, wrapper codegen and device_custom_pass."""
    from torch._inductor.codegen.common import register_backend_for_device

    from pytorchsim.frontend.inductor.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
    from pytorchsim.frontend.inductor.postgrad import pytorchsim_post_grad_pass

    register_backend_for_device("npu", TritonNPUScheduling, TritonNPUWrapperCodegen,
                                device_custom_pass=pytorchsim_post_grad_pass)


# Inductor's device op overrides (register_device_op_overrides)

def _register_device_op_overrides():
    """Register NPUDeviceOpOverrides for npu."""
    from torch._inductor.codegen.common import register_device_op_overrides

    register_device_op_overrides("npu", NPUDeviceOpOverrides())


class NPUDeviceOpOverrides(DeviceOpOverrides):
    """The device code in an npu wrapper: no raw stream, no device to set or sync, a null guard."""

    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx: int) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: int) -> str:
        """A null context: the caller writes `with {this}:`, where "pass" would not parse."""
        return "torch._ops.contextlib.nullcontext()"


# Inductor's choices class (config.inductor_choices_class)

def _register_choices():
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

    from pytorchsim.backend import template_record
    from pytorchsim.frontend.inductor import is_lowering_npu
    from pytorchsim.frontend.inductor.autotune import (generated_kernel_blocks,
                                                       template_kernel_configs)

    class NPUChoices(LookupTableChoices):
        def uuid(self):
            """This class's part of Inductor's cache key: the machine and the picks it hands out."""
            return f"pytorchsim-npu-choices-4-{template_record.graph_digest()}"

        def get_template_configs(self, kernel_inputs, templates, op_name, kwarg_overrides=None):
            """Inductor's choices of `templates`, an npu mm offered the resident mm alone where it
            holds (mm_resident.templates_for), its config marked as planned."""
            from pytorchsim.backend import template_record
            from pytorchsim.frontend.inductor.autotune import mm_resident
            got = super().get_template_configs(
                kernel_inputs, mm_resident.templates_for(kernel_inputs, templates, op_name), op_name,
                kwarg_overrides)
            for c in got:
                ktc = c.annotations.get("ktc")
                if ktc is not None and ktc.template is mm_resident.mm_resident_template:
                    ktc.annotations.setdefault(template_record.SOURCE, template_record.PLANNED)
            return got

        def get_config_heuristics(self, device_type="cuda"):
            if device_type == "npu":
                return template_kernel_configs.conv_config_heuristic()
            return super().get_config_heuristics(device_type)

        def _get_lookup_table(self):
            """This machine's recorded picks; Inductor's own looks only on CUDA."""
            return template_record.lookup_table()

        @staticmethod
        def _get_device_key(device):
            if getattr(device, "type", None) == "npu":
                return template_record.device_key()
            return LookupTableChoices._get_device_key(device)

        def _create_lookup_choices(self, *args, **kwargs):
            """Inductor's choices from the recorded configs, each marked as recorded."""
            ktcs = super()._create_lookup_choices(*args, **kwargs)
            for ktc in ktcs:
                ktc.annotations[template_record.SOURCE] = template_record.RECORDED
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


# Inductor's template heuristics (torch._inductor.heuristics.template.registry)

def _register_template_heuristics():
    """Register the mm/bmm heuristic and its addmm/baddbmm form, and the resident mm's, for npu."""
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.heuristics.template.registry import register_template_heuristic

    from pytorchsim.frontend.inductor.autotune import mm_resident
    from pytorchsim.frontend.inductor.autotune.template_kernel_configs import (
        mm_config_heuristics, mm_resident_config_heuristic)

    mm, addmm = mm_config_heuristics()
    register_template_heuristic(mm_template.uid, "npu")(mm)
    register_template_heuristic(bmm_template.uid, "npu")(mm)
    register_template_heuristic(mm_template.uid, "npu", op_name="addmm")(addmm)
    register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")(addmm)
    register_template_heuristic(mm_resident.mm_resident_template.uid, "npu", op_name="mm")(
        mm_resident_config_heuristic())
