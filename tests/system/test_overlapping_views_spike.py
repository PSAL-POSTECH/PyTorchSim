"""Two kernel arguments that are interleaved views of one storage (K and V out of tokenwise pages, q/k/v out of
one qkv) read their own values on Spike. Breaks when an argument's .raw holds zeros between its elements: the
wrapper copies overlapping .raw files over each other, and the later one zeroes the earlier one's values."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_interleaved_views_of_one_storage():
    torch._dynamo.reset()
    x = torch.randn(1024, 2)
    f = torch.compile(lambda a, b: a * 2 + b, dynamic=False)
    dev = x.to("npu:0")
    got = f(dev[:, 0], dev[:, 1])
    check_result("interleaved views", got.cpu(), x[:, 0] * 2 + x[:, 1])


if __name__ == "__main__":
    test_interleaved_views_of_one_storage()
    print("Test Passed")
