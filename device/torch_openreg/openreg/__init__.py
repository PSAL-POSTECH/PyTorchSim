import os
import threading

import torch
import torch_openreg._C  # type: ignore[misc]

from . import meta  # noqa: F401

_initialized = False
_default_streams = {}
_tog_simulator = None
_launch_context = threading.local()
_eager_compiled = set()

#: Ops with no npu kernel that Inductor can codegen, compiled by default rather than per test.
DEFAULT_EAGER_TO_COMPILE = (
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
    """Context-manager that selects `device` (a torch.device or index; a negative index or None
    is a no-op) and restores the previous one on exit."""

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
    _initialized = True

    num_devices = device_count()
    for device_idx in range(num_devices):
        _default_streams[device_idx] = Stream()

class Stream:
    """Wrapper for OpenReg stream."""

    def __init__(self, flags=0):
        self._stream = torch_openreg._C._stream_create()

    def __del__(self):
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
            pass

    def launch_kernel(self, task):
        """Add a Python callable kernel to this stream."""
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
        _default_streams[device_idx] = Stream()

    return _default_streams[device_idx]


def launch_kernel(tog_path, attribute_path):
    """Queue a launch of `tog_path` on the open TOGSimProcess, on the current device's default
    stream, with the launch context's stream index and timestamp."""
    sim = get_tog_simulator()
    if sim is None:
        raise RuntimeError("[torch.npu] No TOGSim is open. Launch inside `with torch.npu.TOGSimulator(...)`.")

    device_idx = current_device()
    stream_index, timestamp = get_launch_context()
    def launch_task():
        return sim.launch_kernel(device_idx, stream_index, tog_path, attribute_path, timestamp)

    stream = default_stream()
    stream.launch_kernel(launch_task)

def synchronize():
    """Synchronize all streams on the current device: queue the open TOGSimProcess's
    device_synchronize on the default stream, then wait for every task."""
    sim = get_tog_simulator()
    if sim is not None:
        device_idx = current_device()

        def sync_task():
            return sim.device_synchronize(device_idx)

        stream = default_stream()
        stream.launch_kernel(sync_task)

    torch_openreg._C._device_synchronize()

def get_tog_simulator():
    return _tog_simulator

def set_tog_simulator(simulator):
    """Set the global TOGSimProcess instance (or None)."""
    global _tog_simulator
    _tog_simulator = simulator

class TOGSimulator:
    """Open a TOGSim backend and make it this device's simulator for the block, exporting its
    machine file as TORCHSIM_TARGET while the block is open, so kernels compile for the machine TOGSim
    times on."""

    _CONFIG_ENV_UNSET = object()

    def __init__(self, config_path=None, togsim_path=None):
        self.config_path = config_path
        self.togsim_path = togsim_path
        self._backend = None

    def __enter__(self):
        from pytorchsim.simulator.timing.togsim import TOGSimProcess

        self._backend = TOGSimProcess(config_path=self.config_path,
                                      togsim_path=self.togsim_path)
        self._old_config_env = os.environ.get("TORCHSIM_TARGET", self._CONFIG_ENV_UNSET)
        os.environ["TORCHSIM_TARGET"] = os.path.abspath(self._backend.config_path)
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
                os.environ.pop("TORCHSIM_TARGET", None)
            else:
                os.environ["TORCHSIM_TARGET"] = self._old_config_env
        return False


def set_launch_context(stream_index=0, timestamp=0):
    _launch_context.stream_index = stream_index
    _launch_context.timestamp = timestamp

def get_launch_context():
    stream_index = getattr(_launch_context, 'stream_index', 0)
    timestamp = getattr(_launch_context, 'timestamp', 0)
    return stream_index, timestamp

class launch_context:
    """Context manager setting the stream index (partition id) and timestamp (ns) launch_kernel
    uses, e.g. `with torch.npu.launch_context(stream_index=1, timestamp=1000): model(input)`."""

    def __init__(self, stream_index=0, timestamp=0):
        self.stream_index = stream_index
        self.timestamp = timestamp
        self.prev_stream_index = None
        self.prev_timestamp = None

    def __enter__(self):
        self.prev_stream_index = getattr(_launch_context, 'stream_index', 0)
        self.prev_timestamp = getattr(_launch_context, 'timestamp', 0)
        set_launch_context(self.stream_index, self.timestamp)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _launch_context.stream_index = self.prev_stream_index
        _launch_context.timestamp = self.prev_timestamp
        return False

def launch_model(model, *args, stream_index=0, timestamp=0, **kwargs):
    """Run a compiled model on the open TOGSimulator, its kernels launched on `stream_index` at
    `timestamp` ns. Returns the model's output."""
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
    """Run `op` on CPU copies and put the answer back where the caller wants it: the C++
    CPUFallback in Python, since the re-entry it serves must not go through the dispatcher again."""
    cpu_args = _tree_to(args, "cpu")
    cpu_kwargs = {k: _tree_to(v, "cpu") for k, v in kwargs.items()}
    result = op(*cpu_args, **cpu_kwargs)

    schema = getattr(op, "_schema", None)
    outs = [kwargs[a.name] for a in (schema.arguments if schema else ())
            if a.is_out and a.name in kwargs]
    if not outs and kwargs.get("out") is not None:
        out = kwargs["out"]
        outs = [out] if isinstance(out, torch.Tensor) else list(out)
    if outs:
        srcs = [result] if isinstance(result, torch.Tensor) else list(result)
        for dst, src in zip(outs, srcs):
            dst.copy_(src)
        return outs[0] if len(outs) == 1 else tuple(outs)

    device = None
    for x in list(args) + list(kwargs.values()):
        for t in (x if isinstance(x, (list, tuple)) else [x]):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is not None:
            break
    return _tree_to(result, device) if device is not None else result


def _compile_in_flight():
    """Is a torch.compile running on this thread? An eager op reached then is compile-time work
    (a constant Inductor folds), not the workload, and a compile inside a compile breaks dynamo."""
    return torch.compiler.is_compiling()


def eager_to_compile(op_name):
    """Register the eager op `op_name` (e.g. "aten::mul.Tensor") to run as a torch.compile graph.
    An entry while that graph runs, which Inductor's own aten fallback can cause, runs on CPU instead, as
    re-entering would compile the same graph without end."""
    in_flight = threading.local()

    @torch.compiler.disable
    def wrapper(*args, **kwargs):
        namespace, op_path = op_name.split("::", 1)
        op = torch.ops
        for part in [namespace] + op_path.split("."):
            op = getattr(op, part)

        if getattr(in_flight, "busy", False) or _compile_in_flight():
            return _run_on_cpu(op, args, kwargs)

        @torch.compile(dynamic=False)
        def dummy_graph(*args, **kwargs):
            return op(*args, **kwargs)

        in_flight.busy = True
        try:
            return dummy_graph(*args, **kwargs)
        finally:
            in_flight.busy = False

    if op_name in _eager_compiled:
        return
    _eager_compiled.add(op_name)
    torch.library.impl(op_name, "npu", wrapper)

def register_eager_to_compile(ops):
    """Register every op name in `ops` with eager_to_compile."""
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
    "TOGSimulator",
    "eager_to_compile",
    "register_eager_to_compile",
    "DEFAULT_EAGER_TO_COMPILE",
]
