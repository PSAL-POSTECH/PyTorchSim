"""The convolution family: convolutions the conv template takes once rewritten. A depthwise conv2d is
dispatcher/convolution.py, before AOTAutograd."""

import torch
from torch._inductor.pattern_matcher import Arg, CallFunction

from pytorchsim.frontend.inductor.postgrad import register_rewrite

aten = torch.ops.aten


# convolution_transposed: a transposed conv as its direct equivalent

@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))),
                  lambda match: match.args[6] is True)
def convolution_transposed(node, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    """A transposed convolution as its direct equivalent: the input spread by the stride and padded by
    dilation * (kernel - 1) - padding (plus output_padding after), the weight's in and out channels
    swapped per group, flipped, and spread by the dilation; then stride, dilation 1, no padding."""

    def per_dim(seq, n):
        seq = list(seq)
        return seq * n if len(seq) == 1 else seq

    def spread(t, lead, extents, factor):
        """factor - 1 zeros between neighbouring elements along each of the trailing dims."""
        if all(f == 1 for f in factor):
            return t
        n = len(extents)
        paired = list(lead)
        for extent in extents:
            paired += [extent, 1]
        pad = []
        for i in reversed(range(n)):
            pad += [0, factor[i] - 1, 0, 0]
        y = torch.constant_pad_nd(t.reshape(paired), pad, 0.0)
        y = y.reshape(list(lead) + [extents[i] * factor[i] for i in range(n)])
        for i in range(n):
            if factor[i] != 1:
                y = y.narrow(len(lead) + i, 0, extents[i] * factor[i] - (factor[i] - 1))
        return y

    def direct(x, weight, bias):
        batchless = x.dim() == weight.dim() - 1
        if batchless:
            x = x.unsqueeze(0)
        batch, in_chan, *spatial = x.shape
        _, per_group, *kernel = weight.shape
        n = len(kernel)
        out_chan = per_group * groups
        st, pd, dl, op = (per_dim(v, n) for v in (stride, padding, dilation, output_padding))
        y = spread(x, [batch, in_chan], spatial, st)
        pad = []
        for i in reversed(range(n)):
            lo = dl[i] * (kernel[i] - 1) - pd[i]
            pad += [lo, lo + op[i]]
        if any(p != 0 for p in pad):
            y = torch.constant_pad_nd(y, pad, 0.0)
        w = weight.reshape(groups, in_chan // groups, per_group, *kernel).transpose(1, 2)
        w = w.reshape(out_chan, in_chan // groups, *kernel)
        if any(k != 1 for k in kernel):
            w = w.flip(list(range(2, 2 + n)))
        w = spread(w, [out_chan, in_chan // groups], kernel, dl)
        if n == 1:
            out = _conv1d_as_conv2d(y, w, bias, [1], [0], [1], groups)
        else:
            out = torch.convolution(y, w, bias, [1] * n, [0] * n, [1] * n, False, [0] * n, groups)
        return out.squeeze(0) if batchless else out

    return _exact_arity(direct, x, weight, bias)


# convolution_1d: a 1-D conv as a 2-D one over a height of 1

@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))),
                  lambda match: match.args[6] is False and len(match.args[3]) == 1)
def convolution_1d(node, x, weight, bias, stride, padding, dilation, transposed, output_padding,
                   groups):
    """A 1-D convolution as a 2-D one over a height of 1, which the conv template takes; its backward
    graph holds convolution_backward either way, so this needs no place before AOTAutograd."""
    def as_2d(x, weight, bias):
        return _conv1d_as_conv2d(x, weight, bias, stride, padding, dilation, groups)

    return _exact_arity(as_2d, x, weight, bias)


def _conv1d_as_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """conv1d of [N, C, L] as conv2d of [N, C, 1, L]: the height takes stride and dilation 1, no padding."""
    out = torch.convolution(x.unsqueeze(2), weight.unsqueeze(2), bias, [1, stride[0]],
                            [0, padding[0]], [1, dilation[0]], False, [0, 0], groups)
    return out.squeeze(2)


def _exact_arity(fn, x, weight, bias):
    """fn(x, weight, bias) as a replacement taking exactly the operands it is handed: Inductor traces
    every parameter, so with no bias the replacement takes (x, weight) and passes None itself."""
    if bias is None:
        return (lambda x, weight: fn(x, weight, None)), [x, weight]
    return fn, [x, weight, bias]
