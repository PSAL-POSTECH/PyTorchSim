"""npu's DeviceInterface: what Dynamo and Inductor ask of the device, answered for the simulator.
Of it they read the properties, the compute capability, availability, Triton and bf16 support."""

import torch
from torch._dynamo.device_interface import DeviceInterface

from pytorchsim import config


class NPUDeviceProperties:
    """The properties Inductor reads; every other one it reads by getattr with a default."""

    def __init__(self):
        self.multi_processor_count = int(config.CONFIG_MACHINE.vector_lanes)


class NPUDeviceInterface(DeviceInterface):
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
        """None of CUDA's; Triton's compile metadata carries it, which torchsim_compile does not read."""
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


def install():
    """Register NPUDeviceInterface as npu's DeviceInterface."""
    from torch._dynamo.device_interface import register_interface_for_device

    register_interface_for_device("npu", NPUDeviceInterface)
