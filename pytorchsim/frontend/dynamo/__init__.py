"""What npu plugs into Dynamo: its DeviceInterface (device_interface.py)."""


def install():
    """Register npu with Dynamo."""
    from . import device_interface

    device_interface.install()
