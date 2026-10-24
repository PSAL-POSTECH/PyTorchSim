"""What npu plugs into Inductor, in the order Inductor meets it:

    registration.py    everything npu registers, one section per registry: its DeviceInterface, a
                       Triton GPU, its codegen and post-grad pass, its wrappers' device code, its
                       choices and template heuristics
    post_grad_pass.py  the FX graph after Inductor's post-grad passes (device_custom_pass)
    autotune/          how each kernel's sizes are chosen: template kernels' configs and selection,
                       generated kernels' blocks
    codegen/           the scheduling and wrapper Inductor generates through, kernel metadata

is_npu and is_lowering_npu, here, are the device tests the patches share.
"""

import torch

from pytorchsim import config


def install():
    """Register npu with Inductor and apply every patch here, then Inductor's config."""
    from . import autotune, registration

    registration.install()
    autotune.install()
    _configure_inductor()


def _configure_inductor():
    """mm/conv template autotune on, at compile time off. In timing mode a template with an epilogue
    is picked by its fused cycles, Inductor timing the top configs one at a time in this process
    (TritonNPUScheduling.benchmark_codegened_module); template_timing records picks, so its cache
    stays off. No combo kernels: one picks its tensors by program id, which a TOG cannot time, and
    blocks are pinned in triton_kernel_kwargs, a hook combo kernels skip."""
    from torch._inductor import config as inductor_config

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
