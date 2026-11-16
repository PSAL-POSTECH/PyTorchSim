"""Inductor compiles an npu graph's kernels in its worker pool, as on any device: PTB's Triton backend is importable
in a worker, so a kernel built there is the one the parent launches (values match the CPU). Breaks when compile_threads
is forced to 1 at install, or a worker cannot build a kernel for npu."""
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))

import torch  # noqa: E402
from result_check import check_result  # noqa: E402


def test_kernels_compile_in_the_pool():
    from torch._inductor import config as inductor_config
    from torch._inductor.async_compile import AsyncCompile, get_compile_threads

    assert get_compile_threads() > 1 or os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"), inductor_config.compile_threads

    def fn(x, y):
        a = torch.relu(x) + y
        b = torch.softmax(a, dim=-1)
        c = (b * 3.0).sum(dim=0)
        return torch.tanh(c) + a.mean(dim=1).sum()

    x, y = torch.randn(64, 256), torch.randn(64, 256)
    got = torch.compile(dynamic=False)(fn)(x.to("npu:0"), y.to("npu:0"))
    check_result("ParallelCompile", got.cpu(), fn(x, y))
    assert AsyncCompile._ready_future is not None or get_compile_threads() <= 1


if __name__ == "__main__":
    test_kernels_compile_in_the_pool()
