"""Ops the npu takes as a Triton kernel of our own: each module registers its op's npu kernel through
torch.library and launches the kernel as a triton_op (torchsim::<op>). Rewrites into aten ops are
frontend/post_grad_pass.py instead."""


def install():
    """Register every op's npu kernel, once each."""
    from . import grouped_mm, topk
    grouped_mm.install()
    topk.install()
