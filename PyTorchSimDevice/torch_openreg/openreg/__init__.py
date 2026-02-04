import torch
from torch._dynamo.device_interface import register_interface_for_device

import torch_openreg._C  # type: ignore[misc]

from . import meta  # noqa: F401
from . import extension_device_op_overrides
from .extension_device_interface import ExtensionDeviceInterface

_initialized = False
_default_streams = {}  # Dictionary to store default streams per device


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_openreg._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = torch_openreg._C._set_device(self.prev_idx)
        return False


def is_available():
    return True


def device_count() -> int:
    return torch_openreg._C._get_device_count()


def current_device():
    return torch_openreg._C._get_device()


def set_device(device) -> None:
    return torch_openreg._C._set_device(device)

def custom_device():
    return torch.device("npu:0")

def init():
    _lazy_init()


def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    torch_openreg._C._init()
    register_interface_for_device(custom_device(), ExtensionDeviceInterface)
    _initialized = True

    # Create default streams for all devices
    num_devices = device_count()
    for device_idx in range(num_devices):
        _default_streams[device_idx] = Stream()

class Stream:
    """Wrapper for OpenReg stream."""

    def __init__(self, flags=0):
        self._stream = torch_openreg._C._stream_create()

    def __del__(self):
        if hasattr(self, '_stream'):
            torch_openreg._C._stream_destroy(self._stream)

    def launch_kernel(self, task):
        """Add a Python callable kernel to this stream.

        Args:
            task: A Python callable (function) to be executed in the stream
        """
        torch_openreg._C._add_task_to_stream(self._stream, task)

    @property
    def cdata(self):
        """Get the underlying stream pointer (for internal use)."""
        return self._stream


def synchronize():
    """Synchronize all streams on the current device."""
    torch_openreg._C._device_synchronize()


def stream(flags=0):
    """Create a new stream.

    Args:
        flags: Stream flags (optional)

    Returns:
        Stream: A new stream object
    """
    return Stream(flags=flags)

def default_stream(device=None):
    _lazy_init()
    if device is None:
        device_idx = current_device()
    else:
        device_idx = torch.accelerator._get_device_index(device, optional=True)
        if device_idx < 0:
            device_idx = current_device()

    if device_idx not in _default_streams:
        # Create default stream if it doesn't exist
        _default_streams[device_idx] = Stream()

    return _default_streams[device_idx]


def launch_kernel(task, stream=None):
    _lazy_init()
    if stream is None:
        stream = default_stream()
    stream.launch_kernel(task)

from .random import *  # noqa: F403
from .amp import *

__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "custom_device",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "get_amp_supported_dtype",
    "stream",
    "launch_kernel",
    "synchronize",
]
