"""topk is offered a config per candidate XBLOCK (the plan's rows a program and its halves); on tpuv3_32x32 (16 KiB a
lane) 32 rows of a 1024-wide row with K = 32 overflow the scratchpad, the Triton backend raises OutOfResources, Inductor
drops that config and 16 rows run, with the CPU's values and indices. Breaks when topk launches one fixed XBLOCK."""
import os
import sys

os.environ["TORCHSIM_TARGET"] = "tpuv3_32x32"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


@torch.no_grad()
def test_first_candidate_overflows(rows=64, width=1024, k=32):
    from torch._inductor.runtime import triton_heuristics as th
    from triton.runtime.errors import OutOfResources

    outcomes = []
    orig = th.CachingAutotuner._precompile_config

    def spy(self, cfg, **kw):
        if "XBLOCK" not in cfg.kwargs or "R0_BLOCK" in cfg.kwargs:
            return orig(self, cfg, **kw)
        try:
            got = orig(self, cfg, **kw)
        except OutOfResources as e:
            outcomes.append((cfg.kwargs["XBLOCK"], f"OutOfResources: {e}"))
            raise
        outcomes.append((cfg.kwargs["XBLOCK"], "compiled"))
        return got

    torch._dynamo.reset()
    x = torch.randn(rows, width, generator=torch.Generator().manual_seed(0))
    want = torch.topk(x, k)
    orig_patch, th.CachingAutotuner._precompile_config = orig, spy
    try:
        got = torch.compile(lambda t: torch.topk(t, k), dynamic=False)(x.to("npu:0"))
    finally:
        th.CachingAutotuner._precompile_config = orig_patch
    for xb, what in outcomes:
        print(f"candidate XBLOCK={xb}: {what}")
    assert outcomes[0][0] == 32 and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert (16, "compiled") in outcomes, outcomes
    check_result("TopKCandidateOverflow", got[0].cpu(), want.values, 1e-6, 1e-6)
    assert torch.equal(got[1].cpu(), want.indices)


if __name__ == "__main__":
    test_first_candidate_overflows()
