"""A custom kernel (the depthwise conv, a CustomFusedKernel) is offered a config per candidate block from tiling.py;
the first one overflows the scratchpad (tpuv3_32x32's, 1 MiB: v6e's holds the largest tile Triton builds), the Triton backend raises OutOfResources, Inductor drops that config and the
next one runs, with the CPU's values. Breaks when the kernel is launched with one fixed BC, or its configs are not
tried in order."""
import os
import sys
from unittest import mock

os.environ["TORCHSIM_TARGET"] = "tpuv3_32x32"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


@torch.no_grad()
def test_first_candidate_overflows(channels=256, length=512, kernel=4):
    from torch._inductor.runtime import triton_heuristics as th
    from triton.runtime.errors import OutOfResources
    from pytorchsim.frontend import tiling

    outcomes = []
    orig = th.CachingAutotuner._precompile_config

    def spy(self, cfg, **kw):
        if "BC" not in cfg.kwargs:
            return orig(self, cfg, **kw)
        try:
            got = orig(self, cfg, **kw)
        except OutOfResources as e:
            outcomes.append((cfg.kwargs["BC"], f"OutOfResources: {e}"))
            raise
        outcomes.append((cfg.kwargs["BC"], "compiled"))
        return got

    torch._dynamo.reset()
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(channels, channels, kernel, groups=channels, padding=kernel - 1, bias=False)
    x = torch.randn(1, channels, length)
    want = conv(x)[..., :length]
    dev = conv.to("npu:0")
    with mock.patch.object(tiling, "candidate_sizes", lambda first, limit=3: [1 << 10, 128, 64]), \
            mock.patch.object(th.CachingAutotuner, "_precompile_config", spy):
        got = torch.compile(dynamic=False)(lambda t: dev(t)[..., :length])(x.to("npu:0"))
    for bc, what in outcomes:
        print(f"candidate BC={bc}: {what}")
    assert outcomes[0][0] == 1 << 10 and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert (64, "compiled") in outcomes, outcomes
    check_result("DepthwiseCandidateOverflow", got.cpu(), want, 1e-4, 1e-4)


if __name__ == "__main__":
    test_first_candidate_overflows()
