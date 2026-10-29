"""A depthwise 1-D conv (Qwen3.5's and Mamba's causal conv: one channel a group, k = 4, padded k - 1) on
the npu: depthwise_conv's kernel runs it, as conv2d over a height of 1, and the output matches the CPU;
a grouped 1-D conv that is not depthwise still runs and matches."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


@torch.no_grad()
def run(name, channels, length, kernel, groups, dtype=torch.float32, batch=1, depthwise=True):
    """Compile one nn.Conv1d on npu for inference (no gradient, as the model tests run), compare it with the
    CPU, and check whether depthwise_conv ran it."""
    from pytorchsim.frontend.dispatcher.depthwise_conv import DEPTHWISE
    torch._dynamo.reset()
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(channels, channels, kernel, groups=groups, padding=kernel - 1, bias=False).to(dtype)
    x = torch.randn(batch, channels, length, dtype=dtype)
    want = conv(x)[..., :length]
    before = DEPTHWISE.calls.get("_depthwise_kernel", 0)
    dev = conv.to("npu:0")
    got = torch.compile(dynamic=False)(lambda t: dev(t)[..., :length])(x.to("npu:0"))
    tol = (2e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    check_result(name, got.cpu(), want, *tol)
    ran = DEPTHWISE.calls.get("_depthwise_kernel", 0) - before
    assert (ran > 0) == depthwise, f"{name}: depthwise_conv ran {ran} times, depthwise={depthwise}"
    print(f"PASS  {name}: depthwise kernel launches {ran}")


if __name__ == "__main__":
    run("causal conv1d", channels=256, length=128, kernel=4, groups=256)
    run("causal conv1d bf16", channels=256, length=128, kernel=4, groups=256, dtype=torch.bfloat16, batch=2)
    run("grouped conv1d", channels=64, length=32, kernel=3, groups=4, depthwise=False)
    print("Test Passed")
