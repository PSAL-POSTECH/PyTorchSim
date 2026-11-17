"""A cat of two inputs whose pointwise kernel masks each half (Yolov5's triton_poi_fused_add_cat_convolution_silu_5):
numel is a multiple of the block, so Inductor drops the numel mask and only the first half's clamp is left. The grid
covers both halves, so a bound read off that clamp alone refuses the launch."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_cat_of_two_masked_halves(device, shape=(32, 25600)):
    def fn(a, b, c):
        return torch.cat([a + torch.nn.functional.silu(b), torch.nn.functional.silu(c)], dim=0)

    xs = [torch.randn(shape).to(torch.bfloat16).to(device) for _ in range(3)]
    res = torch.compile(dynamic=False)(fn)(*xs)
    check_result("CatMaskedHalves", res, fn(*[x.cpu() for x in xs]), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    test_cat_of_two_masked_halves(torch.device("npu:0"))
