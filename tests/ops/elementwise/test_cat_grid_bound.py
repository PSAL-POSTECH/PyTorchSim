"""A cat of two inputs whose pointwise kernel masks each half (Yolov5's triton_poi_fused_add_cat_convolution_silu_5):
PTB's grid rule reads the first half's clamp (819200 / XBLOCK programs) as the whole grid, so a launch that covers
both halves is refused. BLOCKED on liveness.grid_bound in the compiler, not on a tile choice here."""
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
    check_result("CatMaskedHalves", res, fn(*[x.cpu() for x in xs]))


if __name__ == "__main__":
    test_cat_of_two_masked_halves(torch.device("npu:0"))
