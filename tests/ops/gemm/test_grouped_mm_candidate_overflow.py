"""grouped_mm is offered its ranked (BN, BK) tiles as configs. The tile search is made to rank an oversized tile first
(a 1024 x 1024 weight tile on a 16 KiB-a-lane scratchpad, as an estimate that undercounts would): the Triton backend raises
OutOfResources, Inductor drops that config and the next ranked one runs, with the reference's values. Breaks when the
kernel is launched with the one tile plan ranked first."""
import os

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
import small_machine  # noqa: E402

small_machine.use_machine(16)


@torch.no_grad()
def test_first_tile_overflows(k=1024, n=1024, sizes=(40, 9, 4)):
    from torch._inductor.runtime import triton_heuristics as th
    from triton.runtime.errors import OutOfResources
    from pytorchsim.frontend import tiling

    outcomes = []
    orig = th.CachingAutotuner._precompile_config

    def spy(self, cfg, **kw):
        if "BN" not in cfg.kwargs:
            return orig(self, cfg, **kw)
        try:
            got = orig(self, cfg, **kw)
        except OutOfResources as e:
            outcomes.append((cfg.kwargs["BN"], f"OutOfResources: {e}"))
            raise
        outcomes.append((cfg.kwargs["BN"], "compiled"))
        return got

    real = tiling.gemm_tile_candidates
    greedy = lambda m, n_, k_, c, p: [(m, n_, k_), *real(m, n_, k_, c, p)]
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(sum(sizes), k, generator=g) * 0.1).to(torch.bfloat16)
    b = (torch.randn(len(sizes), k, n, generator=g) * 0.1).to(torch.bfloat16)
    offs = torch.cumsum(torch.tensor(sizes, dtype=torch.int32), 0, dtype=torch.int32)
    want, lo = [], 0
    for gi, hi in enumerate(offs.tolist()):
        want.append(a[lo:hi].float() @ b[gi].float())
        lo = hi
    want = torch.cat(want)
    torch._dynamo.reset()
    with mock.patch.object(tiling, "gemm_tile_candidates", greedy), \
            mock.patch.object(th.CachingAutotuner, "_precompile_config", spy):
        got = torch.compile(lambda *t: torch.nn.functional.grouped_mm(t[0], t[1], offs=t[2]), dynamic=False)(
            a.to("npu:0"), b.to("npu:0"), offs.to("npu:0")).float().cpu()
    for bn, what in outcomes:
        print(f"candidate BN={bn}: {what}")
    assert outcomes[0][0] == n and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert any(w == "compiled" for _, w in outcomes), outcomes
    err = ((got - want).abs().max() / want.abs().max()).item()
    assert err < 2e-2, err
    print("-----------------------------------\n|GroupedMMCandidateOverflow Test Passed|\n-----------------------------------")


if __name__ == "__main__":
    test_first_tile_overflows()
