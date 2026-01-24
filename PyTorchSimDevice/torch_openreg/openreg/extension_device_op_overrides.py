from __future__ import annotations

from textwrap import dedent

from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides

class ExtensionDeviceOpOverrides(DeviceOpOverrides):
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
        return "pass"

register_device_op_overrides("npu", ExtensionDeviceOpOverrides())
register_device_op_overrides("cpu", CpuDeviceOpOverrides())