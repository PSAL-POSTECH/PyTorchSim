"""What npu plugs into Inductor, in the order Inductor meets it:

    registration.py    everything npu registers with Inductor, one section per registry
    postgrad/          the npu's device_custom_pass, one module per op family
    autotune/          how each kernel's sizes are chosen: template configs and selection, generated blocks
    codegen/           the scheduling and wrapper Inductor generates through, kernel metadata

is_npu and is_lowering_npu, here, are the device tests the patches share.
"""

import os

import torch

from pytorchsim import config


def install():
    """Register npu with Inductor and apply every patch here, then Inductor's config."""
    from . import autotune, registration

    registration.install()
    autotune.install()
    _configure_inductor()


def _configure_inductor():
    """mm/conv template autotune on, at compile time off, its tables shown in debug mode unless the env sets
    them; Inductor's autotune cache off (template_timing records picks); no combo kernels, which pick tensors
    by program id (a TOG cannot time that) and skip the triton_kernel_kwargs hook that pins blocks."""
    from torch._inductor import config as inductor_config

    if "TORCHINDUCTOR_AUTOTUNE_NUM_CHOICES_DISPLAYED" not in os.environ:
        inductor_config.autotune_num_choices_displayed = 10 if config.CONFIG_DEBUG_MODE else 0
    if "TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS" not in os.environ:
        inductor_config.max_autotune_report_choices_stats = bool(config.CONFIG_DEBUG_MODE)
    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.max_autotune_conv_backends = "ATEN,TRITON"
    inductor_config.triton.autotune_at_compile_time = False
    timed = bool(config.pytorchsim_timing_mode)
    inductor_config.benchmark_epilogue_fusion = timed
    inductor_config.max_epilogue_benchmarked_choices = config.codegen_autotune_template_topk
    inductor_config.autotune_local_cache = False
    inductor_config.combo_kernels = False
    if timed:
        inductor_config.compile_threads = 1


def is_npu(obj):
    """Whether a device, device string, tensor, layout or node is on this device."""
    dev = getattr(obj, "device", obj)
    dev = getattr(dev, "device", dev)
    if isinstance(dev, str):
        dev = torch.device(dev)
    return getattr(dev, "type", None) == "npu"


def is_lowering_npu():
    """Whether the graph being lowered right now targets this device."""
    from torch._inductor.virtualized import V

    try:
        return V.graph.get_current_device_or_throw().type == "npu"
    except (AttributeError, RuntimeError):
        return False
