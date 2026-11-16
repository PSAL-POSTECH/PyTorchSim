"""Which of a kernel's compiled configs runs: Inductor's autotuner times each through the benchmarker registered
for npu. Functional mode picks the first that compiled, in the order its candidates were offered, without
launching any; timing mode launches each as a measurement, which TOGSim answers in cycles."""

import contextlib
import itertools
import threading

from pytorchsim.backend import template_record

_ORDER = itertools.count(1)
_STATE = threading.local()


def measuring():
    """Whether a launch is being timed by the benchmarker: it then yields its cycles and runs nothing else."""
    return getattr(_STATE, "cycles", None) is not None


def report(cycles):
    """The launch being measured took `cycles` on TOGSim."""
    _STATE.cycles.append(cycles)


def benchmark(benchmarker, fn, *, warmup, rep, **kwargs):
    """The 'time' of fn, in ms: in functional mode later calls are larger, so the first config offered wins
    and fn is not called; in timing mode fn launches the kernel, and TOGSim's cycles (1 cycle = 1 ns) come back,
    inf when it launched nothing (an extern kernel) or did not simulate."""
    from pytorchsim import config
    if not config.pytorchsim_timing_mode:
        template_record.warn_untimed_autotune()
        return next(_ORDER) * 1e-3
    with _measure() as got:
        fn()
    return got[-1] * 1e-6 if got else float("inf")


@contextlib.contextmanager
def _measure():
    _STATE.cycles = []
    try:
        yield _STATE.cycles
    finally:
        _STATE.cycles = None
