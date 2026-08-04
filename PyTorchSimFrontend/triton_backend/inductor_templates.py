"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates themselves are not GPU-specific -- torch ships one
`triton_mm.py.jinja` for cuda, xpu, mtia and cpu -- but `use_triton_template`
gates on `is_gpu`, and GPU_TYPES is a hardcoded list with no registration hook.
"""

import torch


def _register_npu_as_gpu():
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def _claim_triton_present():
    # has_triton() asks whether a supported *device* is available, not whether
    # triton is installed. The missing piece is a driver we never use.
    import torch.utils._triton as triton_utils
    import torch._inductor.scheduler as scheduler

    triton_utils.has_triton = lambda: True
    if hasattr(scheduler, "has_triton"):
        scheduler.has_triton = lambda: True


def _register_template_heuristics():
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.template_heuristics.registry import (
        register_template_heuristic)
    from torch._inductor.template_heuristics.triton import (
        BaseConfigHeuristic, MMTemplateConfigMixin)

    @register_template_heuristic(mm_template.uid, "npu")
    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        # TODO: size these from the hardware config (lanes, spad per lane)
        # rather than taking the generic set.
        def __init__(self):
            super().__init__()
            self.exhaustive_configs = self.mm_configs


def pick_config(choices):
    """Stand in for benchmarking: there is no device to time on.

    TODO: rank by simulated cycles. `timing.run_togsim` already returns a cycle
    count per compiled kernel; a real implementation drives each candidate
    through tnpu and caches the result per (kernel, config). Until then the
    offered order wins -- deterministic, and not a claim about speed.
    """
    return {c: 1.0 + i * 1e-3 for i, c in enumerate(choices)}


def _install_selection():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        return pick_config(choices)

    # Precompiling builds every candidate for the current GPU. We need only the
    # chosen kernel's source; tnpu compiles it ahead of time.
    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.make_precompile_fn = lambda self, *a, **k: (lambda: None)


_installed = False


def install():
    global _installed
    if _installed:
        return
    from torch._inductor import config

    _register_npu_as_gpu()
    _claim_triton_present()
    _register_template_heuristics()
    _install_selection()

    # Templates are only considered under max_autotune; selection is replaced
    # above, so this costs a rendering rather than a benchmark sweep.
    config.max_autotune = True
    config.max_autotune_gemm_backends = "TRITON"
    config.max_autotune_conv_backends = "TRITON"
    _installed = True
