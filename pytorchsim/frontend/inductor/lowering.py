"""Inductor lowering patches for npu that Inductor has no hook for: a convolution's groups recorded for
the conv heuristic, and a grouped conv's grid per group. A transposed or 1-D conv and a zero-length
GEMM are rewritten before lowering (post_grad_pass.py); a depthwise conv2d before AOTAutograd
(dispatcher/convolution.py)."""

import torch

from pytorchsim.frontend.inductor.autotune.template_kernel_configs import conv_groups
from pytorchsim.frontend.inductor.registration import is_lowering_npu


def install():
    """Apply this file's patches."""
    _record_conv_groups()
    _size_grouped_conv_grid_per_group()


def _record_conv_groups():
    """Wrap aten.convolution's lowering to record its groups, which narrow the conv heuristic's BLOCK_N
    to one group's channels (choices.conv_groups_now)."""
    import functools
    import inspect

    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel

    sig = inspect.signature(conv_kernel.convolution)

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            groups = bound.arguments.get("groups", 1)
            prev = getattr(conv_groups, "value", 1)
            conv_groups.value = groups if isinstance(groups, int) else 1
            try:
                return inner(*args, **kwargs)
            finally:
                conv_groups.value = prev

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_grouped_conv_grid_per_group():
    """Grid a grouped convolution over one group's channels, not all of them."""
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    orig2d = conv_kernel.conv2d_grid
    orig3d = conv_kernel.conv3d_grid

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return orig2d(n, c, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        if not is_lowering_npu():
            return orig3d(n, c, d, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_grid = conv2d_grid
    conv_kernel.conv3d_grid = conv3d_grid
    for tmpl, fn in ((getattr(conv_kernel, "conv2d_template", None), conv2d_grid),
                     (getattr(conv_kernel, "conv3d_template", None), conv3d_grid)):
        if tmpl is not None:
            tmpl.grid = fn
