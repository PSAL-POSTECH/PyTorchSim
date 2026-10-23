"""How a template's config is picked for npu: TOGSim cycles in timing mode (backend/template_timing.py), the
offered order otherwise, a recorded pick when there is one. Inductor's recording hooks
(add_preprocessing_fn, add_feedback_saver) take the last; it has none for timing a choice, so the
rest replaces four AlgorithmSelectorCache methods, each falling back to Inductor's off npu.
"""

import torch

from pytorchsim import config
from pytorchsim.frontend.inductor import is_npu


def install():
    """Patch algorithm selection for npu choices: cycles from TOGSim in timing mode, the offered
    order otherwise; no precompile; allocate-only examples. A recorded pick is the only choice,
    and a timed one is recorded."""
    from torch._inductor import config as inductor_config
    from torch._inductor import select_algorithm
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    from pytorchsim.backend import template_timing

    orig_benchmark = AlgorithmSelectorCache.__dict__["benchmark_choices"].__func__
    orig_call = AlgorithmSelectorCache.__call__
    orig_precompile = AlgorithmSelectorCache.make_precompile_fn
    orig_example = AlgorithmSelectorCache.__dict__["generate_example_value"].__func__

    def _choices_are_npu(choices):
        return any(is_npu(getattr(c, "layout", None)) for c in choices)

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        if not _choices_are_npu(choices):
            return orig_benchmark(cls, choices, autotune_args, is_collective)
        timed = bool(config.pytorchsim_timing_mode)
        if timed != inductor_config.benchmark_epilogue_fusion:
            raise RuntimeError(
                "TORCHSIM_TIMING_MODE changed after pytorchsim was installed: Inductor's "
                "epilogue-fusion benchmark follows the mode the process started in")
        if timed and template_timing.times(choices):
            return template_timing.time_choices(choices)
        return pick_config(choices)

    def __call__(self, name, choices, *args, **kwargs):
        """Inductor's selection; a fusion is benchmarked only for choices timed in cycles."""
        if _choices_are_npu(choices) and not template_timing.times(choices):
            kwargs["return_multi_template"] = False
        return orig_call(self, name, choices, *args, **kwargs)

    def make_precompile_fn(self, choices, *args, **kwargs):
        if not _choices_are_npu(choices):
            return orig_precompile(self, choices, *args, **kwargs)
        return lambda: None

    def generate_example_value(size, stride, device, dtype, extra_size,
                               allocation_size=None):
        """An unread benchmark operand, allocated rather than sampled."""
        if not is_npu(device):
            return orig_example(size, stride, device, dtype, extra_size,
                                allocation_size)
        shape = size if allocation_size is None else allocation_size
        needed = extra_size
        if all(s > 0 for s in shape):
            needed += sum((s - 1) * t for s, t in zip(shape, stride)) + 1
        view = torch.as_strided(
            torch.empty(needed, dtype=dtype, device=device), shape, stride)
        return view if tuple(shape) == tuple(size) else view.as_strided(size, stride)

    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.__call__ = __call__
    AlgorithmSelectorCache.make_precompile_fn = make_precompile_fn
    AlgorithmSelectorCache.generate_example_value = staticmethod(
        generate_example_value)
    select_algorithm.add_preprocessing_fn(template_timing.keep_recorded)
    select_algorithm.add_feedback_saver(template_timing.record)


def pick_config(choices):
    """Timings in place of benchmarking: the offered order wins, extern last."""
    from torch._inductor.select_algorithm import ExternKernelCaller

    return {c: (1e3 if isinstance(c, ExternKernelCaller) else 1.0) + i * 1e-3
            for i, c in enumerate(choices)}
