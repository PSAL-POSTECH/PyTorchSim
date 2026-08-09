"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates themselves are not GPU-specific -- torch ships one
`triton_mm.py.jinja` for cuda, xpu, mtia and cpu -- but `use_triton_template`
gates on `is_gpu`, and GPU_TYPES is a hardcoded list with no registration hook.
"""

import os

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
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.template_heuristics.registry import (
        register_template_heuristic)
    from torch._inductor.template_heuristics.triton import (
        AddMMConfigMixin, BaseConfigHeuristic, MMTemplateConfigMixin)

    @register_template_heuristic(mm_template.uid, "npu")
    @register_template_heuristic(bmm_template.uid, "npu")
    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        # TODO: size these from the hardware config (lanes, spad per lane)
        # rather than taking the generic set.
        def __init__(self):
            super().__init__()
            self.exhaustive_configs = self.mm_configs

    # addmm and baddbmm carry a bias as input_nodes[0]; without their own entry
    # the mm heuristic is used with prefix_args=0 and def_kernel asserts.
    @register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        pass


def _size_conv_blocks_from_the_machine():
    """Offer conv tiles this machine has lanes for.

    `get_config_heuristics` has no registry lookup -- it is an if/elif over
    cuda, xpu, cpu, mtia and then `BaseConfigHeuristic()` -- so npu takes the
    generic set, whose first entry is ConvConfig(64, 256, 16, 2, 4). With 128
    lanes that is a [64, 256] tile banked on the N axis, two columns per lane,
    and bank_vectorize refuses it:

      operand [64, 256] banked on axis 1 with strides [1, 64] is not reached by
      indexing its own row-major type in the nest [2, 64] this op's maps ask for

    resnet18 reaches it at its eleventh conv, the first whose out_chan is 256 --
    below that `preprocess_mm_configs` clamps BLOCK_N to the channel count and
    the tile happens to fit.

    THIS IS THE MAPPING POLICY, NOT A WORKAROUND FOR THAT REFUSAL. A block size
    is a statement about the machine the kernel runs on, and taking a table
    written for a GPU is not one -- the heuristic registered below has carried
    the same TODO since it was written. The refusal is a real defect and stays
    one: a tile deeper than one element per lane is a shape this backend has to
    handle, and pinning BLOCK_N to the lane count only stops resnet from being
    the thing that reports it.

    N IS THE LANE AXIS, so it takes the lane count exactly -- the same choice
    kernel_spec.fixed_config_for makes for XBLOCK, and for the same reason. M
    and K are per-lane depth and cost scratchpad rather than lanes, so they are
    offered small-to-large and the first that the shape does not clamp wins.
    """
    from torch._inductor.choices import InductorChoices
    from torch._inductor.template_heuristics.triton import (
        BaseConfigHeuristic, ConvConfig)

    from PyTorchSimFrontend import extension_config

    lanes = int(extension_config.vpu_num_lanes)

    class NPUConfigHeuristic(BaseConfigHeuristic):
        def __init__(self):
            super().__init__()
            self.conv_configs = [
                ConvConfig(64, lanes, 16, 1, 4),
                ConvConfig(32, lanes, 16, 1, 4),
                ConvConfig(64, lanes, 32, 1, 4),
            ]

    original = InductorChoices.get_config_heuristics

    def get_config_heuristics(self, device_type="cuda"):
        if device_type == "npu":
            return NPUConfigHeuristic()
        return original(self, device_type)

    InductorChoices.get_config_heuristics = get_config_heuristics


def pick_config(choices):
    """Stand in for benchmarking: there is no device to time on, so the offered
    order wins. Extern ranks last, present only so a device with no registered
    heuristic (cpu) still has a choice.

    TODO: rank by simulated cycles; timing.run_togsim already returns one per
    compiled kernel.
    """
    from torch._inductor.select_algorithm import ExternKernelCaller

    return {c: (1e3 if isinstance(c, ExternKernelCaller) else 1.0) + i * 1e-3
            for i, c in enumerate(choices)}


def _short_circuit_degenerate_gemms():
    """A zero-length axis has no tile, so the heuristics offer no config and the
    empty choice list raises. A MoE expert routing no tokens gives [0, K] @ [K, N].
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.lowering import full, lowerings
    from torch._inductor.virtualized import V

    def wrap(op, bias):
        def wrapped(*args, _orig=lowerings[op], **kwargs):
            try:
                m, n, k, layout = mm_args(*args[bias:bias + 2],
                                          layout=kwargs.get("layout"))[:4]
                m, n, k = (int(V.graph.sizevars.size_hint(s)) for s in (m, n, k))
            except Exception:  # noqa: BLE001 - dynamic shape; leave it to _orig
                return _orig(*args, **kwargs)
            # k == 0 sums nothing, so zeros -- except addmm/baddbmm, which are
            # then beta * bias.
            if m == 0 or n == 0 or (k == 0 and not bias):
                return full(layout.size, 0, dtype=layout.dtype,
                            device=layout.device)
            return _orig(*args, **kwargs)

        return wrapped

    aten = torch.ops.aten
    for op, bias in ((aten.mm, 0), (aten.bmm, 0),
                     (aten.addmm, 1), (aten.baddbmm, 1)):
        for name in op.overloads():
            o = getattr(op, name)
            if o in lowerings:
                lowerings[o] = wrap(o, bias)


def _install_selection():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        return pick_config(choices)

    # Precompiling builds every candidate for the current GPU; we need only the
    # chosen kernel's source.
    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.make_precompile_fn = lambda self, *a, **k: (lambda: None)


_installed = False


def install():
    """On by default; TORCHSIM_TRITON_TEMPLATES=0 opts out. Sending mm to aten
    simulates nothing, so a test that stops inside tnpu says more than one that
    passes without running the op.
    """
    global _installed
    if _installed or os.environ.get("TORCHSIM_TRITON_TEMPLATES", "1") == "0":
        return
    from torch._inductor import config

    _register_npu_as_gpu()
    _claim_triton_present()
    _register_template_heuristics()
    _size_conv_blocks_from_the_machine()
    _short_circuit_degenerate_gemms()
    _install_selection()

    # Not max_autotune: that also turns on pointwise autotuning, which appends
    # a benchmark harness to every kernel module.
    config.max_autotune_gemm = True
    # These are global but the heuristics are registered for npu only, so ATEN
    # stays in the list to keep a cpu gemm in the same graph from having no
    # choice at all. pick_config ranks it last.
    config.max_autotune_gemm_backends = "ATEN,TRITON"
    config.max_autotune_conv_backends = "ATEN,TRITON"
    config.triton.autotune_at_compile_time = False
    # Epilogue-fusion benchmarking renders a benchmark-flavoured kernel whose
    # harness imports land indented in the real module.
    config.benchmark_epilogue_fusion = False
    _installed = True
