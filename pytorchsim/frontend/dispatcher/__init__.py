"""The npu kernels of aten ops, registered through torch.library above autograd: ops the npu takes
as a Triton kernel of our own, each a CustomFusedKernel (custom_fused_kernel.py), and conv2d's
depthwise stencil (convolution.py). Rewrites after AOTAutograd are inductor/postgrad/."""


def install():
    """Register every op's npu kernel, once each."""
    from pytorchsim import config
    from . import attention, attention_paged_kvcache, convolution, grouped_mm, grouped_mm_xla_structure, sort, topk
    attention.install(attention_paged_kvcache.PAGED_SDPA if config.CONFIG_VALIDATION_KVCACHE_LAYOUT == "paged" else None)
    convolution.install()
    (grouped_mm_xla_structure if "megablox" in config.CONFIG_VALIDATION_XLA_STRUCTURE else grouped_mm).install()
    sort.install()
    topk.install()
