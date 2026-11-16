"""How a template's config is picked for npu: Inductor's own selection over the choices that compiled, the npu's
benchmarker (backend/config_pick.py) ranking them by TOGSim cycles in timing mode and in offered order otherwise;
a recorded pick is the only choice (Inductor's recording hooks). Inductor has no hook for an unread operand or
for fusing only choices timed in cycles, so three AlgorithmSelectorCache methods are wrapped, each Inductor's off npu."""

import torch

from pytorchsim import config
from pytorchsim.frontend.inductor import is_npu


def install():
    """Patch algorithm selection for npu choices: allocate-only examples; no multi-template buffer unless timed.
    A recorded pick is the only choice, and a timed one is recorded."""
    from torch._inductor import config as inductor_config
    from torch._inductor import select_algorithm
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, NoValidChoicesError

    from pytorchsim.backend import template_timing
    from pytorchsim.backend import template_record

    orig_benchmark = AlgorithmSelectorCache.__dict__["benchmark_choices"].__func__
    orig_call = AlgorithmSelectorCache.__call__
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
        if not timed:
            template_record.warn_untimed_autotune()
        timings = orig_benchmark(cls, choices, autotune_args, is_collective)
        if timed and template_timing.times(choices):
            for c in choices:
                if template_record.is_timed(c):
                    c.annotations["ktc"].annotations[template_record.SOURCE] = template_record.TIMED
        return timings

    def __call__(self, name, choices, *args, **kwargs):
        """Inductor's selection; a fusion is benchmarked only for choices timed in cycles."""
        if _choices_are_npu(choices) and not template_timing.times(choices):
            kwargs["return_multi_template"] = False
        try:
            return orig_call(self, name, choices, *args, **kwargs)
        except NoValidChoicesError:
            _refuse_external(choices)
            raise

    def _refuse_external(choices):
        """An external mapping's tile is the only candidate: its failure to compile is an error naming the
        mapping file and the entry, with why."""
        for c in choices:
            ktc = getattr(c, "annotations", {}).get("ktc")
            if ktc is None or ktc.annotations.get(template_record.SOURCE) != template_record.EXTERNAL:
                continue
            m, n, k = ktc.inputs.mnk_hinted()
            try:
                c.precompile()
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{config.codegen_external_mapping_file}: the entry {m}_{n}_{k} -> "
                                 f"{c.description} does not compile on this machine: {e}") from e

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
    AlgorithmSelectorCache.generate_example_value = staticmethod(
        generate_example_value)
    select_algorithm.add_preprocessing_fn(template_record.keep_recorded)
    select_algorithm.add_feedback_saver(template_record.record)
