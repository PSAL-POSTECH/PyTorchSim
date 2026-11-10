"""A graph compiled once over a dynamic size runs on Spike at every size: the launcher builds the ELF for
extents it was not built for (each tensor's span, each numel passed as a scalar). Breaks when a launch
reuses the import-time ELF, which Spike's placement refuses (ShapeMismatch). Values match the CPU."""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_one_compile_runs_every_size():
    f = torch.compile(lambda x, y: x * 2 + y, dynamic=False)
    for i, n in enumerate((1024, 1536, 640)):
        x, y = torch.randn(4, n), torch.randn(4, n)
        dev = [t.to("npu:0") for t in (x, y)]
        for t in dev:
            torch._dynamo.mark_dynamic(t, 1)
        with torch.compiler.set_stance("fail_on_recompile") if i else torch.compiler.set_stance("default"):
            got = f(*dev)
        check_result(f"n={n}", got.cpu(), x * 2 + y)


if __name__ == "__main__":
    test_one_compile_runs_every_size()
    print("Test Passed")
