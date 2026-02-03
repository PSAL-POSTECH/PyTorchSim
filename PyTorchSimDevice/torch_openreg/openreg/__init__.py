import torch
from torch._dynamo.device_interface import register_interface_for_device

import torch_openreg._C  # type: ignore[misc]

from . import meta  # noqa: F401
from . import extension_device_op_overrides
from .extension_device_interface import ExtensionDeviceInterface

_initialized = False


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


class Stream:
    """Wrapper for OpenReg stream."""

    def __init__(self, priority=None, flags=0):
        if priority is not None:
            self._stream = torch_openreg._C._stream_create_with_priority(flags, priority)
        else:
            self._stream = torch_openreg._C._stream_create()

    def __del__(self):
        if hasattr(self, '_stream'):
            torch_openreg._C._stream_destroy(self._stream)

    def synchronize(self):
        """Wait for all operations in the stream to complete."""
        torch_openreg._C._stream_synchronize(self._stream)

    def query(self):
        """Check if all operations in the stream have completed."""
        return torch_openreg._C._stream_query(self._stream)

    def wait_event(self, event):
        """Make this stream wait for an event."""
        torch_openreg._C._stream_wait_event(self._stream, event._event)

    def get_priority(self):
        """Get the priority of the stream."""
        return torch_openreg._C._stream_get_priority(self._stream)

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


class Event:
    """Wrapper for OpenReg event."""

    def __init__(self, enable_timing=False):
        if enable_timing:
            # orEventEnableTiming = 0x1
            self._event = torch_openreg._C._event_create_with_flags(0x1)
        else:
            self._event = torch_openreg._C._event_create()

    def __del__(self):
        if hasattr(self, '_event'):
            torch_openreg._C._event_destroy(self._event)

    def record(self, stream=None):
        """Record the event in a stream."""
        if stream is None:
            # Use default stream (stream 0)
            stream = Stream()
        torch_openreg._C._event_record(self._event, stream._stream)

    def synchronize(self):
        """Wait for the event to complete."""
        torch_openreg._C._event_synchronize(self._event)

    def query(self):
        """Check if the event has completed."""
        return torch_openreg._C._event_query(self._event)

    def elapsed_time(self, start_event):
        """Get the elapsed time between two events in milliseconds."""
        return torch_openreg._C._event_elapsed_time(start_event._event, self._event)

    @property
    def cdata(self):
        """Get the underlying event pointer (for internal use)."""
        return self._event


def synchronize():
    """Synchronize all streams on the current device."""
    torch_openreg._C._device_synchronize()


def stream(priority=None, flags=0):
    """Create a new stream.

    Args:
        priority: Stream priority (optional)
        flags: Stream flags (optional)

    Returns:
        Stream: A new stream object
    """
    return Stream(priority=priority, flags=flags)


def event(enable_timing=False):
    """Create a new event.

    Args:
        enable_timing: Whether to enable timing for the event

    Returns:
        Event: A new event object
    """
    return Event(enable_timing=enable_timing)


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
    "Stream",
    "Event",
    "stream",
    "event",
    "synchronize",
]
