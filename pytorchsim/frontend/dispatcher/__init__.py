"""The npu kernels of aten ops, registered through torch.library above autograd: ops the npu takes
as a Triton kernel of our own, each a CustomFusedKernel (custom_fused_kernel.py), and conv2d's
depthwise stencil (convolution.py). Rewrites after AOTAutograd are inductor/post_grad_pass/."""


def install():
    """Register every op's npu kernel, once each."""
    from . import attention, convolution, grouped_mm, topk
    attention.install()
    convolution.install()
    grouped_mm.install()
    topk.install()
