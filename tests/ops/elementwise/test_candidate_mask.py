"""Every tile a generated kernel is offered must be one Inductor's source is right for: Inductor drops an axis's
mask when numel divides its max block, so a candidate larger than that max block, not dividing numel, runs the
last program past the tensor. Breaks if a candidate outgrows TritonNPUKernel.max_block (MobileNet's 401408 and
Yolov5's 3 x 409600 channels-last copy fault or are refused). A y axis past one y grid keeps its mask; reading
it as dropped left Diffusion's bf16 channels-last silu (ynumel 102400) with no tile and a None YBLOCK."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_pointwise_numel_multiple_of_4096(device, shape=(1, 32, 112, 112)):
    def fn(a, b):
        return (a * b + 1.5).clamp(0, 6)

    x = torch.randn(shape).to(device)
    y = torch.randn(shape).to(device)
    res = torch.compile(dynamic=False)(fn)(x, y)
    check_result("PointwiseMaskElided", res, fn(x.cpu(), y.cpu()))


def test_channels_last_copy(device, shape=(1, 3, 640, 640)):
    def fn(a):
        return (a + 1).contiguous(memory_format=torch.channels_last)

    x = torch.randn(shape).to(device)
    res = torch.compile(dynamic=False)(fn)(x)
    check_result("ChannelsLastCopy", res, fn(x.cpu()))


def test_y_past_one_grid_keeps_its_mask(device, shape=(1, 9, 320, 320)):
    def fn(a):
        return torch.nn.functional.silu(a).contiguous()

    x = torch.randn(shape, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    res = torch.compile(dynamic=False)(fn)(x.to(device))
    check_result("YPastOneGrid", res, fn(x))


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pointwise_numel_multiple_of_4096(device)
    test_channels_last_copy(device)
    test_y_past_one_grid_keeps_its_mask(device)
