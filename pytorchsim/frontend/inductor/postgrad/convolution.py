"""The convolution family: convolutions rewritten into what the conv or mm template takes. A depthwise conv2d is
dispatcher/convolution.py, before AOTAutograd."""

import torch
from torch._inductor.pattern_matcher import Arg, CallFunction

from pytorchsim.frontend.inductor.postgrad import fake_value, register_rewrite

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


# convolution_pointwise: a 1x1 conv as the matrix its weight already is

def _is_pointwise(match):
    """A floating 2-D conv, not transposed, of one group, a 1x1 kernel at stride 1, no padding or dilation."""
    x, weight, _, stride, padding, dilation, transposed, _, groups = match.args
    xv, wv = fake_value(x), fake_value(weight)
    if transposed is not False or groups != 1 or xv is None or wv is None or xv.dim() != 4:
        return False
    return (list(wv.shape[2:]) == [1, 1] and all(s == 1 for s in stride) and all(p == 0 for p in padding)
            and all(d == 1 for d in dilation) and xv.is_floating_point())


@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))), _is_pointwise)
def convolution_pointwise(node, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    """A 1x1 convolution as [O, C] times the pixels, [C, N * H * W]: the weight is already that matrix and the
    product is already the output's rows, so neither operand is transposed and the lanes ride the pixels --
    the conv template would put them on the output channels, which a stem has sixteen of."""
    def pointwise(x, weight, bias):
        n, c, h, w = x.shape
        o = weight.shape[0]
        rows = weight.reshape(o, c)
        y = (torch.mm(rows, x.reshape(c, h * w)) if n == 1
             else torch.matmul(rows, x.reshape(n, c, h * w)))
        if bias is not None:
            y = y + bias.reshape(o, 1)
        return y.reshape(n, o, h, w)

    return _exact_arity(pointwise, x, weight, bias)


# convolution_patches: a conv whose kernel is its stride (a ViT's patch embedding) as one mm

def _is_patches(match):
    """A floating 2-D conv, not transposed, of one group, no padding or dilation, kernel = stride > 1."""
    x, weight, _, stride, padding, dilation, transposed, _, groups = match.args
    xv, wv = fake_value(x), fake_value(weight)
    if transposed is not False or groups != 1 or xv is None or wv is None or xv.dim() != 4:
        return False
    kernel = list(wv.shape[2:])
    return (len(stride) == 2 and list(stride) == kernel and kernel != [1, 1] and xv.is_floating_point()
            and all(p == 0 for p in padding) and all(d == 1 for d in dilation))


@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))), _is_patches)
def convolution_patches(node, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    """Each kernel-sized patch as a row, [N * Ho * Wo, C * kh * kw], times the weight's [O, C * kh * kw]
    transposed: one mm, where the conv template would loop over every tap at K = C."""
    def patches(x, weight, bias):
        n, c, h, w = x.shape
        o, _, kh, kw = weight.shape
        ho, wo = h // kh, w // kw
        x = x[:, :, :ho * kh, :wo * kw]
        rows = x.reshape(n, c, ho, kh, wo, kw).permute(0, 2, 4, 1, 3, 5).reshape(n * ho * wo, c * kh * kw)
        cols = weight.reshape(o, c * kh * kw).t()
        y = torch.mm(rows, cols) if bias is None else torch.addmm(bias, rows, cols)
        return y.reshape(n, ho, wo, o).permute(0, 3, 1, 2)

    return _exact_arity(patches, x, weight, bias)


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
