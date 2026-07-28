import torch
from torch._dynamo.device_interface import DeviceInterface, caching_worker_current_devices, caching_worker_device_properties

class _ExtensionDeviceProperties:   # FIXME: Dummy property values
    name: str = "Extension_device"
    platform_name: str
    vendor: str
    driver_version: str
    version: str
    max_compute_units: int
    gpu_eu_count: int
    max_work_group_size: int
    max_num_sub_groups: int
    sub_group_sizes: list[int]
    has_fp16: bool
    has_fp64: bool
    has_atomic64: bool
    has_bfloat16_conversions: bool
    has_subgroup_matrix_multiply_accumulate: bool
    has_subgroup_matrix_multiply_accumulate_tensor_float32: bool
    has_subgroup_2d_block_io: bool
    total_memory: int
    multi_processor_count: int = 128     # gpu_subslice_count, num_sm
    architecture: int
    type: str

_ExtensionDeviceProperties = _ExtensionDeviceProperties

class ExtensionDeviceInterface(DeviceInterface):
    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["extension_device"] = device

        @staticmethod
        def current_device() -> int:
            if "extension_device" in caching_worker_current_devices:
                return caching_worker_current_devices["extension_device"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> _ExtensionDeviceProperties:
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "extension_device"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = ExtensionDeviceInterface.Worker.current_device()

            if "extension_device" not in caching_worker_device_properties:
                device_prop = [
                    torch.cuda.get_device_properties(i)
                    for i in range(torch.cuda.device_count())
                ]
                caching_worker_device_properties["extension_device"] = device_prop

            return _ExtensionDeviceProperties

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None):
        return 36