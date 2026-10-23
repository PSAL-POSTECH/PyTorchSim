"""What npu plugs into Inductor, in the order Inductor meets it:

    registration.py    npu as an Inductor backend: GPU_TYPES, Triton present, codegen and op overrides
    device_interface.py  what Inductor asks of the device: properties (SM count), Triton, bf16
    device_op_overrides.py  the device code Inductor writes into an npu wrapper
    post_grad_pass.py  the FX graph after Inductor's post-grad passes (device_custom_pass)
    lowering.py        lowering patches with no Inductor hook, npu's conv decompositions
    choices.py         NPUChoices: conv tiles, pinned blocks, reductions, recorded template picks
    selection.py       how a template's config is picked (TOGSim cycles in timing mode)
    codegen/           the scheduling and wrapper Inductor generates through, kernel metadata, tiling
"""

from pytorchsim import config


def install():
    """Register npu with Inductor and apply every patch here, then Inductor's config."""
    from . import choices, lowering, registration, selection

    registration.install()
    choices.install()
    lowering.install()
    selection.install()
    _configure_inductor()


def _configure_inductor():
    """mm/conv template autotune on, at compile time off. In timing mode a template with an epilogue
    is picked by its fused cycles, Inductor timing the top configs one at a time in this process
    (TritonNPUScheduling.benchmark_codegened_module); select records picks, so its cache stays off."""
    from torch._inductor import config as inductor_config

    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.max_autotune_conv_backends = "ATEN,TRITON"
    inductor_config.triton.autotune_at_compile_time = False
    timed = bool(config.pytorchsim_timing_mode)
    inductor_config.benchmark_epilogue_fusion = timed
    inductor_config.max_epilogue_benchmarked_choices = config.codegen_autotune_template_topk
    inductor_config.autotune_local_cache = False
    if timed:
        inductor_config.compile_threads = 1
