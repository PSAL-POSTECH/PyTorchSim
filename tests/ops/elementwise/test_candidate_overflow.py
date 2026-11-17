"""A generated kernel whose first candidate tile overflows the scratchpad is dropped by Inductor's autotuner
(OutOfResources from the Triton backend) and the next candidate is the one that runs, with the right values.
Breaks if the kernel's configs are pinned to one tile, or the backend's overflow is not OutOfResources."""
import os
import sys

import torch
from unittest import mock

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_first_candidate_overflows(device, numel=1 << 20):
    from torch._inductor.runtime import triton_heuristics as th
    from triton.runtime.errors import OutOfResources
    from pytorchsim.frontend import tiling

    outcomes = []
    orig = th.CachingAutotuner._precompile_config

    def spy(self, cfg, **kw):
        try:
            got = orig(self, cfg, **kw)
        except OutOfResources as e:
            outcomes.append((cfg.kwargs["XBLOCK"], f"OutOfResources: {e}"))
            raise
        outcomes.append((cfg.kwargs["XBLOCK"], "compiled"))
        return got

    def fn(*xs):
        out = xs[0]
        for t in xs[1:]:
            out = out * 0.5 + t
        return out

    xs = [torch.randn(numel).to(device) for _ in range(32)]
    with mock.patch.object(tiling, "candidate_blocks", lambda *a, **k: [{"XBLOCK": numel}, {"XBLOCK": 4096}]), \
            mock.patch.object(th.CachingAutotuner, "_precompile_config", spy):
        res = torch.compile(dynamic=False)(fn)(*xs)
    for xblock, what in outcomes:
        print(f"candidate XBLOCK={xblock}: {what}")
    assert outcomes[0][0] == numel and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert (4096, "compiled") in outcomes, outcomes
    check_result("CandidateOverflow", res, fn(*[t.cpu() for t in xs]))


if __name__ == "__main__":
    test_first_candidate_overflows(torch.device("npu:0"))
