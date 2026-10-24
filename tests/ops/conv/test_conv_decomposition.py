"""Convolutions the npu runs as other ops, against the CPU: a depthwise conv2d as a stencil on the
vector unit (dispatcher/convolution.py, before AOTAutograd, so its backward is the stencil's), a 1-D
conv as a 2-D one (postgrad.convolution_1d); and no npu entry in Inductor's global
decomposition table.

  conv1d               Conv1d, stride 2, padding 1
  conv1d_same          Conv1d, padding="same"
  depthwise            depthwise conv2d, stride 2, padding 1
  depthwise_same       depthwise conv2d, padding="same" with an even kernel (asymmetric)
  depthwise_batchless  depthwise conv2d on a [C, H, W] input
  grouped              a conv2d of two groups that is not depthwise: aten's own convolution
  depthwise_grad       compiled, input and weight requiring grad: gradients match, and no
                       convolution_backward is left to aten (it would run on the CPU)
  depthwise_inference  compiled under inference_mode
  no_global_decomp     aten.convolution is not in Inductor's decomposition table

Usage: test_conv_decomposition.py [case ...]   (all cases when none is named)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")

#  name: (input shape, weight shape, stride, padding, groups)
CASES = {
    "conv1d": ((2, 8, 32), (16, 8, 3), 2, 1, 1),
    "conv1d_same": ((2, 8, 32), (16, 8, 3), 1, "same", 1),
    "depthwise": ((2, 16, 15, 15), (16, 1, 3, 3), 2, 1, 16),
    "depthwise_same": ((1, 16, 12, 12), (16, 1, 4, 4), 1, "same", 16),
    "depthwise_batchless": ((16, 12, 12), (16, 1, 3, 3), 1, 1, 16),
    "grouped": ((1, 8, 12, 12), (16, 4, 3, 3), 1, 1, 2),
}


def _conv(x, w, b, stride, padding, groups):
    fn = F.conv1d if w.dim() == 3 else F.conv2d
    return fn(x, w, b, stride=stride, padding=padding, groups=groups)


def _inputs(name):
    xs, ws, *_ = CASES[name]
    g = torch.Generator().manual_seed(0)
    return torch.randn(*xs, generator=g), torch.randn(*ws, generator=g), torch.randn(ws[0], generator=g)


def run_case(name):
    x, w, b = _inputs(name)
    params = CASES[name][2:]
    torch._dynamo.reset()
    f = torch.compile(lambda x, w, b: _conv(x, w, b, *params), dynamic=False)
    with torch.no_grad():
        out = f(x.to(device), w.to(device), b.to(device))
    check_result(f"ConvDecomposition {name}", out, _conv(x, w, b, *params), rtol=1e-3, atol=1e-3)


def run_grad(name):
    from pytorchsim.backend import codecache
    externs = []
    codecache.observe_every_extern(externs.append)
    x, w, b = _inputs("depthwise")
    params = CASES["depthwise"][2:]
    xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
    _conv(xr, wr, b, *params).sum().backward()
    xd, wd = x.to(device).requires_grad_(), w.to(device).requires_grad_()
    torch._dynamo.reset()
    torch.compile(lambda x, w: _conv(x, w, b.to(device), *params), dynamic=False)(xd, wd).sum().backward()
    check_result("ConvDecomposition depthwise_grad (input)", xd.grad, xr.grad, rtol=1e-3, atol=1e-3)
    check_result("ConvDecomposition depthwise_grad (weight)", wd.grad, wr.grad, rtol=1e-3, atol=1e-3)
    left = [e for e in externs if "convolution" in str(e)]
    check_result("ConvDecomposition depthwise_grad (no aten convolution)", torch.tensor([not left]),
                 torch.tensor([True]))


def run_inference(name):
    x, w, b = _inputs("depthwise")
    params = CASES["depthwise"][2:]
    torch._dynamo.reset()
    f = torch.compile(lambda x, w, b: _conv(x, w, b, *params), dynamic=False)
    with torch.inference_mode():
        out = f(x.to(device), w.to(device), b.to(device))
    check_result("ConvDecomposition depthwise_inference", out, _conv(x, w, b, *params),
                 rtol=1e-3, atol=1e-3)


def run_no_global_decomp(name):
    from torch._inductor.decomposition import select_decomp_table
    check_result("ConvDecomposition no_global_decomp",
                 torch.tensor([torch.ops.aten.convolution.default not in select_decomp_table()]),
                 torch.tensor([True]))


EXTRA = {"depthwise_grad": run_grad, "depthwise_inference": run_inference,
         "no_global_decomp": run_no_global_decomp}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    for n in names:
        (EXTRA[n] if n in EXTRA else run_case)(n)
