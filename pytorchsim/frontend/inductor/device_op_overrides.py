"""The device code Inductor writes into an npu wrapper: no raw stream, no device to set or sync, a
null device guard; cpu keeps its own."""

from __future__ import annotations

from textwrap import dedent

from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides

class NPUDeviceOpOverrides(DeviceOpOverrides):
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

def install():
    """Register npu's device op overrides, and cpu's own beside them."""
    register_device_op_overrides("npu", NPUDeviceOpOverrides())
    register_device_op_overrides("cpu", CpuDeviceOpOverrides())