"""Ops the npu takes as a Triton kernel of our own, each a CustomFusedKernel (fused_kernel.py): the
ops it takes, its plan, its launch as a triton_op (torchsim::<op>) and an aten reference. Rewrites
into aten ops are inductor/post_grad_pass.py instead."""


def install():
    """Register every op's npu kernel, once each."""
    from . import attention, grouped_mm, topk
    attention.install()
    grouped_mm.install()
    topk.install()
