"""The device code Inductor writes into an npu wrapper, in its device op overrides
(torch._inductor.codegen.common.register_device_op_overrides): no raw stream, no device to set or
sync, a null device guard. Inductor registers its own devices' itself."""

from textwrap import dedent

from torch._inductor.codegen.common import DeviceOpOverrides


def install():
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
