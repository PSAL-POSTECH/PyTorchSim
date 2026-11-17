"""Device-side runtime ops called directly, outside any compiled graph: the device synchronizes when torch asks
for every stream, and eager abs covers the dtypes models hand it, contiguous or not."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


def test_synchronize_device():
    """torch.accelerator.synchronize() reaches the guard's synchronizeDevice, which torch refuses by default."""
    torch.accelerator.synchronize()
    torch.accelerator.synchronize(0)
    torch.npu.synchronize()
    check_result("SynchronizeDevice", torch.ones(1), torch.ones(1))


def test_abs_dtypes_and_layouts():
    """Eager abs equals the CPU's for every dtype the stub dispatches, on a contiguous tensor, a transpose
    and a strided slice; a float-only kernel refuses half and bf16 (TritonBench fp8_gemm)."""
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64):
        base = (torch.randn(6, 10, generator=g) * 50).to(dtype)
        for name, view in (("contig", lambda t: t), ("transpose", lambda t: t.t()),
                           ("strided", lambda t: t[:, ::2])):
            out = torch.abs(view(base.to("npu"))).cpu()
            assert out.dtype == dtype, (dtype, name, out.dtype)
            check_result(f"Abs {dtype} {name}", out, torch.abs(view(base)), rtol=0, atol=0)


if __name__ == "__main__":
    test_synchronize_device()
    test_abs_dtypes_and_layouts()
