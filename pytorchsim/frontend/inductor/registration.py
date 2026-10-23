"""Everything npu registers with Inductor: its DeviceInterface, a Triton GPU (GPU_TYPES, which
has no hook, so it patches module state), its codegen and post-grad pass, and the device code its
wrappers carry. is_npu and lowering_npu are the device tests patches share."""

from textwrap import dedent

import torch
from torch._dynamo.device_interface import DeviceInterface
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides

from pytorchsim import config


def install():
    """Make Inductor take npu as a Triton GPU and register npu's device interface, codegen,
    post-grad pass and op overrides."""
    _register_device_interface()
    _register_npu_as_gpu()
    _register_backend()
    _register_device_op_overrides()


def _register_backend():
    """npu's codegen route: Inductor's Triton codegen through TritonNPUScheduling and its wrapper, with
    pytorchsim_post_grad_pass run on every npu graph."""
    from torch._inductor.codegen.common import register_backend_for_device

    from pytorchsim.frontend.inductor.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
    from pytorchsim.frontend.inductor.post_grad_pass import pytorchsim_post_grad_pass

    register_backend_for_device("npu", TritonNPUScheduling, TritonNPUWrapperCodegen,
                                device_custom_pass=pytorchsim_post_grad_pass)


def _register_device_interface():
    """What Inductor asks of the device: its properties, Triton and bf16 (NPUDeviceInterface)."""
    from torch._dynamo.device_interface import register_interface_for_device

    register_interface_for_device("npu", NPUDeviceInterface)


def _register_device_op_overrides():
    """The device code Inductor writes into an npu wrapper, and cpu's own beside it."""
    from torch._inductor.codegen.common import register_device_op_overrides

    register_device_op_overrides("npu", NPUDeviceOpOverrides())
    register_device_op_overrides("cpu", CpuDeviceOpOverrides())


def _register_npu_as_gpu():
    """Add npu to Inductor's GPU_TYPES, so Triton codegen is chosen for it."""
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def lowering_npu():
    """Whether the graph being lowered right now targets this device."""
    from torch._inductor.virtualized import V

    try:
        return V.graph.get_current_device_or_throw().type == "npu"
    except Exception:  # noqa: BLE001
        return False


def is_npu(obj):
    """Whether a device, device string, tensor, layout or node is on this device."""
    dev = getattr(obj, "device", obj)
    dev = getattr(dev, "device", dev)
    if isinstance(dev, str):
        return dev.split(":")[0] == "npu"
    return getattr(dev, "type", None) == "npu"


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
