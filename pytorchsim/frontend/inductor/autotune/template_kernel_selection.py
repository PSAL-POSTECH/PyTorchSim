"""How a template's config is picked for npu: Inductor's own selection over the choices that compiled, the npu's
benchmarker (backend/config_pick.py) ranking them by TOGSim cycles in timing mode and in offered order otherwise;
a recorded pick is the only choice. Registered through select_algorithm's preprocessing and feedback hooks; Inductor
has no hook for an unread operand, so generate_example_value (select_algorithm.py, called by benchmark_example_value)
alone is replaced, Inductor's off npu."""

import torch

from pytorchsim import config
from pytorchsim.frontend.inductor import is_npu


def install():
    """Register the selection hooks; allocate-only examples for npu operands."""
    from torch._inductor import select_algorithm
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    from pytorchsim.backend import template_record

    orig_example = AlgorithmSelectorCache.__dict__["generate_example_value"].__func__

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

    AlgorithmSelectorCache.generate_example_value = staticmethod(
        generate_example_value)
    select_algorithm.add_preprocessing_fn(template_record.keep_recorded)
    select_algorithm.add_preprocessing_fn(template_record.refuse_external_that_does_not_compile)
    select_algorithm.add_feedback_saver(template_record.mark_timed)
    select_algorithm.add_feedback_saver(template_record.record)
