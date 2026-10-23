"""Inductor lowering patches for npu that Inductor has no hook for: a convolution's groups recorded for
the conv heuristic, a grouped conv's grid per group; and npu's conv decompositions. A transposed conv
and a zero-length GEMM are rewritten before lowering, in post_grad_pass.py."""

import torch

from pytorchsim.frontend.inductor.choices import conv_groups
from pytorchsim.frontend.inductor.registration import is_lowering_npu


def install():
    """Apply this file's patches."""
    _record_conv_groups()
    _size_grouped_conv_grid_per_group()
    _decompose_convolution_for_npu()


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


def _decompose_convolution_for_npu():
    """Register npu's aten.convolution decompositions: a Conv1d reaches the 2-D template through
    torch's conv1d_to_conv2d, and a depthwise 2-D conv is a stencil on the vector unit. Declines
    non-npu inputs, transposed convs and output_padding; anything else keeps the template."""
    from torch._inductor.decomposition import conv1d_to_conv2d, register_decomposition

    aten = torch.ops.aten

    @register_decomposition([aten.convolution])
    def _npu_convolution(input, weight, bias, stride, padding, dilation,
                         transposed, output_padding, groups):
        if input.device.type != "npu" or transposed:
            return NotImplemented
        if any(p != 0 for p in output_padding):
            return NotImplemented
        if len(stride) == 1:
            return conv1d_to_conv2d(input, weight, bias, stride, padding, dilation, groups)
        if len(stride) == 2 and _is_depthwise(input, weight, groups):
            return _depthwise_stencil(input, weight, bias, stride, padding, dilation)
        return NotImplemented


def _is_depthwise(input, weight, groups):
    """One channel a group, in and out: the conv is a per-channel stencil, not a matmul."""
    return (groups > 1 and input.shape[1] == groups and weight.shape[0] == groups
            and weight.shape[1] == 1)


def _depthwise_stencil(x, w, bias, stride, padding, dilation):
    """A depthwise 2-D conv as taps: each a strided slice of the padded input times its weight, summed."""
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    g, _, kh, kw = w.shape
    x = torch.ops.aten.constant_pad_nd(x, [pw, pw, ph, ph])
    h_out = (x.shape[2] - dh * (kh - 1) - 1) // sh + 1
    w_out = (x.shape[3] - dw * (kw - 1) - 1) // sw + 1
    out = None
    for i in range(kh):
        for j in range(kw):
            tap = x[:, :, i * dh:i * dh + sh * (h_out - 1) + 1:sh,
                    j * dw:j * dw + sw * (w_out - 1) + 1:sw]
            term = tap * w[:, 0, i, j].reshape(1, g, 1, 1)
            out = term if out is None else out + term
    return out if bias is None else out + bias.reshape(1, g, 1, 1)
