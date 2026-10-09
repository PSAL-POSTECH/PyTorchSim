import os
import threading

import torch
from torch._dynamo.device_interface import register_interface_for_device
import torch_openreg._C  # type: ignore[misc]

from . import meta  # noqa: F401
from . import extension_device_op_overrides
from .extension_device_interface import ExtensionDeviceInterface

_initialized = False
_default_streams = {}  # Dictionary to store default streams per device
_tog_simulator = None  # Singleton TOGSimulator instance
_launch_context = threading.local() # storage for launch_kernel context
_eager_compiled = set()  # op names already given an eager_to_compile wrapper

# Ops with no npu kernel that Inductor can codegen. Registering them here rather
# than per test is what keeps `every operation runs on the NPU` true by default.
DEFAULT_EAGER_TO_COMPILE = (
    "aten::fill_.Scalar",
    "aten::zero_",
    "aten::cat.out",
    "aten::mm.out",
    "aten::max_pool2d_with_indices.out",
    "aten::upsample_nearest2d.out",
    "aten::argmax.out",
    "aten::argmin.out",
    "aten::max.dim_max",
    "aten::min.dim_min",
)

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
    global _initialized, _tog_simulator
    if is_initialized():
        return

    torch_openreg._C._init()
    register_interface_for_device(custom_device(), ExtensionDeviceInterface)
    _initialized = True

    # Set default SDPA backend to math-only for this device.
    torch._C._set_sdp_use_flash(False)
    torch._C._set_sdp_use_overrideable(False)
    torch._C._set_sdp_use_math(True)

    # Create default streams for all devices
    num_devices = device_count()
    for device_idx in range(num_devices):
        _default_streams[device_idx] = Stream()

class Stream:
    """Wrapper for OpenReg stream."""

    def __init__(self, flags=0):
        self._stream = torch_openreg._C._stream_create()

    def __del__(self):
        # Interpreter shutdown can clear module globals before __del__ runs.
        # Only destroy when both runtime handle and stream are still valid.
        stream = getattr(self, "_stream", None)
        backend = globals().get("torch_openreg", None)
        c_api = getattr(backend, "_C", None) if backend is not None else None
        if stream is None or c_api is None:
            return
        destroy = getattr(c_api, "_stream_destroy", None)
        if destroy is None:
            return
        try:
            destroy(stream)
        except (AttributeError, TypeError):
            # Ignore cleanup-time teardown ordering issues.
            pass

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

def stream(flags=0):
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


def launch_kernel(tog_path, attribute_path):
    """Launch a kernel on TOGSimulator.

    Args:
        tog_path: Path to TOG file
        attribute_path: Path to attribute file

    Returns:
        int: The kernel ID assigned to this launch

    """
    # Get TOGSimulator instance
    sim = get_tog_simulator()
    if sim is None:
        raise RuntimeError("[torch.npu] TOGSimulator is not initialized. Call torch.npu.init() first.")

    device_idx = current_device()
    stream_index, timestamp = get_launch_context()
    # Create a task function that calls TOGSimulator.launch_kernel
    def launch_task():
        return sim.launch_kernel(device_idx, stream_index, tog_path, attribute_path, timestamp)

    stream = default_stream()
    stream.launch_kernel(launch_task)

def synchronize():
    """Synchronize all streams on the current device.

    This function:
    1. Registers TOGSimulator.device_synchronize as a task on the default stream
    2. Calls the underlying device_synchronize to wait for all tasks to complete
    """
    # Get TOGSimulator instance
    sim = get_tog_simulator()
    if sim is not None:
        # Get current device index
        device_idx = current_device()

        # Create a task function that calls TOGSimulator.device_synchronize
        def sync_task():
            return sim.device_synchronize(device_idx)

        # Register as task on default stream
        stream = default_stream()
        stream.launch_kernel(sync_task)

    # Call underlying device_synchronize to wait for all tasks to complete
    torch_openreg._C._device_synchronize()

def get_tog_simulator():
    return _tog_simulator

def set_tog_simulator(simulator):
    """Set the global TOGSimulator instance.

    Args:
        simulator: TOGSimulator instance or None
    """
    global _tog_simulator
    _tog_simulator = simulator

class simulator:
    """Open a TOGSim backend and make it this device's simulator for the block.

    Compilation must read the same hardware YAML as the run, so the config path
    is exported while the block is open and restored on exit.
    """

    _CONFIG_ENV_UNSET = object()

    def __init__(self, config_path=None, togsim_path=None):
        self.config_path = config_path
        self.togsim_path = togsim_path
        self._backend = None

    def __enter__(self):
        from Simulator.simulator import TOGSimulator

        self._backend = TOGSimulator(config_path=self.config_path,
                                     togsim_path=self.togsim_path)
        self._old_config_env = os.environ.get("TOGSIM_CONFIG", self._CONFIG_ENV_UNSET)
        os.environ["TOGSIM_CONFIG"] = os.path.abspath(self._backend.config_path)
        self._old_simulator = get_tog_simulator()
        set_tog_simulator(self._backend)
        return self._backend

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Drain this device's stream first, then shut the backend down."""
        try:
            synchronize()
            self._backend.shutdown()
        finally:
            set_tog_simulator(self._old_simulator)
            if self._old_config_env is self._CONFIG_ENV_UNSET:
                os.environ.pop("TOGSIM_CONFIG", None)
            else:
                os.environ["TOGSIM_CONFIG"] = self._old_config_env
        return False


def set_launch_context(stream_index=0, timestamp=0):
    _launch_context.stream_index = stream_index
    _launch_context.timestamp = timestamp

def get_launch_context():
    stream_index = getattr(_launch_context, 'stream_index', 0)
    timestamp = getattr(_launch_context, 'timestamp', 0)
    return stream_index, timestamp

class launch_context:
    """Context manager for setting launch_kernel parameters.

    Args:
        stream_index: Stream index (partition ID) to use for launch_kernel
        timestamp: Timestamp in nanoseconds to use for launch_kernel

    Example:
        with torch.npu.launch_context(stream_index=1, timestamp=1000):
            model(input)
    """

    def __init__(self, stream_index=0, timestamp=0):
        self.stream_index = stream_index
        self.timestamp = timestamp
        self.prev_stream_index = None
        self.prev_timestamp = None

    def __enter__(self):
        # Save previous context values
        self.prev_stream_index = getattr(_launch_context, 'stream_index', 0)
        self.prev_timestamp = getattr(_launch_context, 'timestamp', 0)
        # Set new context values
        set_launch_context(self.stream_index, self.timestamp)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context values
        _launch_context.stream_index = self.prev_stream_index
        _launch_context.timestamp = self.prev_timestamp
        return False

def launch_model(model, *args, stream_index=0, timestamp=0, **kwargs):
    """Launch a compiled model on TOGSimulator.

    Args:
        model: Compiled model (torch.compile())
        *args: Model input arguments
        stream_index: Stream index (partition ID). If None, uses context value.
        timestamp: Timestamp in nanoseconds. If None, uses context value.
        **kwargs: Additional keyword arguments for model execution

    Returns:
        Model output (same as calling model(*args, **kwargs))

    Note:
        This function executes the compiled model and automatically launches
        the generated kernels with the specified stream_index and timestamp.
        If stream_index or timestamp are not provided, values from the current
        context (set via launch_context() or set_launch_context()) are used.
    """
    # Get stream_index and timestamp from parameters or context
    with launch_context(stream_index=stream_index, timestamp=timestamp):
        return model(*args, **kwargs)

from .random import *  # noqa: F403
from .amp import *

def _tree_to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_to(v, device) for v in x)
    return x


def _run_on_cpu(op, args, kwargs):
    """Run `op` on CPU copies and put the answer back where the caller wants it.

    This is what the C++ CPUFallback does for an op the device has no kernel
    for, written here in Python because the re-entry below has to reach it
    WITHOUT going through the dispatcher again -- going through the dispatcher
    is the thing that loops.
    """
    cpu_args = _tree_to(args, "cpu")
    cpu_kwargs = {k: _tree_to(v, "cpu") for k, v in kwargs.items()}
    result = op(*cpu_args, **cpu_kwargs)

    out = kwargs.get("out")
    if out is not None:
        # An out= variant must write the caller's tensor and return it, not a
        # copy of it: the graph that called it already holds that buffer.
        outs = [out] if isinstance(out, torch.Tensor) else list(out)
        srcs = [result] if isinstance(result, torch.Tensor) else list(result)
        for dst, src in zip(outs, srcs):
            dst.copy_(src)
        return out

    device = None
    for x in list(args) + list(kwargs.values()):
        for t in (x if isinstance(x, (list, tuple)) else [x]):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is not None:
            break
    return _tree_to(result, device) if device is not None else result


def eager_to_compile(op_name):
    """
    Register an eager mode operation as a graph-based implementation using torch.compile().

    Args:
        op_name: Operator name (e.g., "aten::mul.Tensor")

    Example:
        torch.npu.eager_to_compile("aten::mul.Tensor")

    RE-ENTRY IS NOT RECURSION HERE, IT IS A LOOP THAT CANNOT END. The compiled
    graph for an op is allowed to contain that same op again -- Inductor emits
    an aten fallback for anything it cannot codegen, and for an out= variant the
    dispatcher reaches it through the functional one -- so the registered impl
    calls torch.compile, which produces a graph that calls the registered impl.
    Nothing about the second trip differs from the first, so it runs until
    Python's stack ends.

        measured   torch.cat over two complex64 tensors under torch.compile
                   with aten::cat.out registered:
                   RecursionError: maximum recursion depth exceeded.
                   Inductor declines complex ("Torchinductor does not support
                   code generation for complex operators"), emits
                   aten.cat.default, and cat.default's out-variant path lands
                   back on this wrapper. Kimi-VL's MoonViT builds its 2D rope
                   table exactly that way. Without the registration the same
                   program runs; the registration is what closes the loop.

    So the first entry compiles, and an entry that arrives while that graph is
    still running executes on CPU instead. That is the same answer the device
    gives for any op it has no kernel for -- what is lost is the compiling of
    the inner op, which was never possible anyway.
    """
    in_flight = threading.local()

    def wrapper(*args, **kwargs):
        # Convert "aten::mul.Tensor" -> torch.ops.aten.mul.Tensor
        namespace, op_path = op_name.split("::", 1)
        op = torch.ops
        for part in [namespace] + op_path.split("."):
            op = getattr(op, part)

        if getattr(in_flight, "busy", False):
            return _run_on_cpu(op, args, kwargs)

        @torch.compile(dynamic=False)
        def dummy_graph(*args, **kwargs):
            return op(*args, **kwargs)

        from torch._dynamo.variables.torch_function import (
            torch_function_mode_stack_state_mgr as modes)
        outer = modes.stack
        in_flight.busy = True
        try:
            return dummy_graph(*args, **kwargs)
        finally:
            in_flight.busy = False
            modes.stack = outer

    if op_name in _eager_compiled:
        return
    _eager_compiled.add(op_name)
    torch.library.impl(op_name, "npu", wrapper)

def register_eager_to_compile(ops):
    """
    Register multiple operators at once using eager_to_compile.

    Args:
        ops: List of operator names (e.g., ["aten::mul.Tensor", "aten::add.Tensor"])

    Example:
        torch.npu.register_eager_to_compile(["aten::mul.Tensor", "aten::add.Tensor"])
    """
    for op_name in ops:
        eager_to_compile(op_name)

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
    "launch_model",
    "synchronize",
    "get_tog_simulator",
    "set_tog_simulator",
    "simulator",
    "eager_to_compile",
    "register_eager_to_compile",
    "DEFAULT_EAGER_TO_COMPILE",
]
