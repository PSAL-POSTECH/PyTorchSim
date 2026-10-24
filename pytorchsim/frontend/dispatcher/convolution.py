"""aten::conv2d on the npu: a depthwise conv (one channel a group) as a stencil on the vector unit, any
other as aten's own convolution. Registered through torch.library above autograd, so a compiled
call is decomposed before AOTAutograd and its backward is the stencil's, on the device."""

import torch

aten = torch.ops.aten

_lib = None


def install():
    """Register aten::conv2d's npu kernel, both overloads, above autograd and at the backend key."""
    global _lib
    if _lib is None:
        _lib = torch.library.Library("aten", "IMPL")
        for key in ("AutogradPrivateUse1", "PrivateUse1"):
            _lib.impl("conv2d", conv2d, key)
            _lib.impl("conv2d.padding", conv2d, key)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """aten's conv2d (a [C, H, W] input batched as one), with a string padding made explicit; a
    depthwise one is _depthwise_stencil, any other aten.convolution."""
    stride, dilation = _pair(stride), _pair(dilation)
    batched = input.dim() == 4
    x = input if batched else input.unsqueeze(0)
    if isinstance(padding, str):
        x, padding = _explicit_padding(x, weight, padding, stride, dilation)
    padding = _pair(padding)
    if _is_depthwise(x, weight, groups):
        out = _depthwise_stencil(x, weight, bias, stride, padding, dilation)
    else:
        out = aten.convolution.default(x, weight, bias, stride, padding, dilation, False, [0, 0],
                                       groups)
    return out if batched else out.squeeze(0)


def _pair(v):
    return [int(v)] * 2 if isinstance(v, int) else [int(e) for e in v]


def _explicit_padding(x, weight, padding, stride, dilation):
    """(x, padding) for padding "valid" or "same": "same" pads dilation * (kernel - 1) per dim, the
    odd one on the high side as aten does, so the input is padded first when they differ."""
    if padding == "valid":
        return x, [0, 0]
    if padding != "same":
        raise ValueError(f"Invalid padding string '{padding}', should be one of 'valid', 'same'")
    if any(s != 1 for s in stride):
        raise ValueError("padding='same' is not supported for strided convolutions")
    total = [d * (k - 1) for d, k in zip(dilation, weight.shape[2:])]
    if all(t % 2 == 0 for t in total):
        return x, [t // 2 for t in total]
    x = aten.constant_pad_nd.default(x, [0, total[1] % 2, 0, total[0] % 2], 0.0)
    return x, [t // 2 for t in total]


def _is_depthwise(x, weight, groups):
    """One channel a group, in and out: the conv is a per-channel stencil, not a matmul."""
    return (groups > 1 and x.shape[1] == groups and weight.shape[0] == groups
            and weight.shape[1] == 1)


def _depthwise_stencil(x, w, bias, stride, padding, dilation):
    """A depthwise 2-D conv as taps: each a strided slice of the padded input times its weight,
    summed."""
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    g, _, kh, kw = w.shape
    x = aten.constant_pad_nd.default(x, [pw, pw, ph, ph])
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
