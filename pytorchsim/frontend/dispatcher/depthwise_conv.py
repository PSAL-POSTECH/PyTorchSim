"""A depthwise 2-D convolution (one channel a group) as one Triton kernel on the vector unit: a program is
BC channels of one output row, the taps a loop in it, each tap one load of the input row it reads, so the
kernel reads the same whatever the kernel size. Its fallback is the taps as aten ops (convolution.py)."""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)


class DepthwiseConv2d(CustomFusedKernel):
    name = "torchsim::depthwise_conv2d"
    #: conv2d's npu kernel (convolution.py) routes its depthwise calls here; it registers no op itself.
    takes = ()

    def kernels(self):
        return (_depthwise_kernel,)

    def plan(self, x, w, bias, stride, padding, dilation):
        """BC channels a program, the lanes, halved while a row's tiles overrun the scratchpad budget."""
        if not x.is_floating_point() or x.dtype != w.dtype:
            return Decline(f"input {x.dtype}, weight {w.dtype}")
        wo = _out_size(x.shape[3], w.shape[3], stride[1], padding[1], dilation[1])
        if wo <= 0 or _out_size(x.shape[2], w.shape[2], stride[0], padding[0], dilation[0]) <= 0:
            return Decline("an empty output")
        bw = next_pow2(wo)
        bc = int(config.CONFIG_MACHINE.vector_lanes)
        while bc > 1 and not fits_scratchpad(_tiles(bc, bw, x.element_size())):
            bc //= 2
        if not fits_scratchpad(_tiles(bc, bw, x.element_size())):
            return Decline(f"an output row of {wo} overruns the scratchpad budget")
        return {"BC": bc, "BW": bw}

    def launch(self, plan, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], stride: list[int],
               padding: list[int], dilation: list[int]) -> torch.Tensor:
        n, c, h, wd = x.shape
        kh, kw = w.shape[2], w.shape[3]
        ho = _out_size(h, kh, stride[0], padding[0], dilation[0])
        wo = _out_size(wd, kw, stride[1], padding[1], dilation[1])
        out = x.new_empty(n, c, ho, wo)
        b = bias if bias is not None else w.new_zeros(c)
        wrap_triton(_depthwise_kernel)[(n * triton.cdiv(c, plan["BC"]), ho, 1)](
            x.contiguous(), w.contiguous(), b, out, c, h, wd, ho, wo, kh, kw, stride[0], stride[1],
            padding[0], padding[1], dilation[0], dilation[1], plan["BC"], plan["BW"])
        return out

    def fallback(self, x, w, bias, stride, padding, dilation):
        from pytorchsim.frontend.dispatcher.convolution import depthwise_taps
        return depthwise_taps(x, w, bias, stride, padding, dilation)


DEPTHWISE = DepthwiseConv2d()


def install():
    DEPTHWISE.install()


def _out_size(size, k, s, p, d):
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def _tiles(bc, bw, esize):
    """[rows, cols, bytes] of one program: an input row, the fp32 sum, the output row, a tap's weights."""
    return [(bc, bw, esize), (bc, bw, 4), (bc, bw, esize), (bc, 1, esize)]


@triton.jit
def _depthwise_kernel(x_ptr, w_ptr, b_ptr, o_ptr, C, H: tl.constexpr, W: tl.constexpr, HO: tl.constexpr,
                      WO: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr,
                      SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr,
                      DW: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    """Output row oh of BC channels of image n: for each tap (i, j), the input row it reads, shifted by
    j * DW, times the channels' weight of that tap, summed in fp32; a row or column past the input is 0."""
    pid = tl.program_id(0)
    oh = tl.program_id(1)
    cblocks = tl.cdiv(C, BC)
    n = pid // cblocks
    cc = (pid % cblocks) * BC + tl.arange(0, BC)
    ow = tl.arange(0, BW)
    live_c = cc < C
    live_w = ow < WO
    acc = tl.zeros((BC, BW), dtype=tl.float32)
    for i in range(KH):
        ih = oh * SH + i * DH - PH
        for j in range(KW):
            iw = ow * SW + j * DW - PW
            ok = live_c[:, None] & (live_w & (iw >= 0) & (iw < W))[None, :] & (ih >= 0) & (ih < H)
            xv = tl.load(x_ptr + (n * C + cc[:, None]) * (H * W) + ih * W + iw[None, :], mask=ok, other=0.0)
            wv = tl.load(w_ptr + cc * (KH * KW) + i * KW + j, mask=live_c, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)[:, None]
    acc += tl.load(b_ptr + cc, mask=live_c, other=0.0).to(tl.float32)[:, None]
    tl.store(o_ptr + (n * C + cc[:, None]) * (HO * WO) + oh * WO + ow[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=live_c[:, None] & live_w[None, :])
