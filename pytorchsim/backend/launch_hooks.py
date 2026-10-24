"""What can watch a launch: a callback per user-defined Triton kernel, at every launch, with every
launch's inputs, and at every call Inductor leaves to aten (the generated wrapper's torchsim_extern)."""

_LAUNCH_OBSERVERS = {}
_EVERY_LAUNCH = []
_EVERY_LAUNCH_INPUTS = []
_EVERY_EXTERN = []


def user_kernel_id(kernel):
    """A user-defined Triton kernel's name in its launch meta: its module and qualified name."""
    return f"{kernel.__module__}.{kernel.__qualname__}"


def observe_launches(kernel, callback):
    """Call `callback()` at every launch of the user-defined Triton kernel `kernel`."""
    _LAUNCH_OBSERVERS[user_kernel_id(kernel)] = callback


def observe_every_launch(callback):
    """Call `callback(launcher)` at every launch of any kernel."""
    _EVERY_LAUNCH.append(callback)


def observe_every_launch_inputs(callback):
    """Call `callback(launcher, args)` at every launch of any kernel, before it runs: `args` are the
    launch's own arguments, the tensors holding what the kernel will read."""
    _EVERY_LAUNCH_INPUTS.append(callback)


def observe_every_extern(callback):
    """Call `callback(call)` at every call Inductor leaves to aten (an extern kernel or a fallback):
    `call` is its op, e.g. "aten.sort.stable". It runs as aten does, not on the simulator."""
    _EVERY_EXTERN.append(callback)


def torchsim_extern(call):
    """The generated wrapper's note before an extern call; see observe_every_extern."""
    for callback in _EVERY_EXTERN:
        callback(call)


def notify_launch(launcher, args):
    """Call, before `launcher` runs: its user-defined kernel's observer, every launch's, and every
    launch's with its inputs."""
    observer = _LAUNCH_OBSERVERS.get(launcher.meta.get("user_kernel"))
    if observer is not None:
        observer()
    for callback in _EVERY_LAUNCH:
        callback(launcher)
    for callback in _EVERY_LAUNCH_INPUTS:
        callback(launcher, args)
