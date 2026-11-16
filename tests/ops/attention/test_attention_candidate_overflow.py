"""The fused attention kernels are offered a config per candidate block (the plan's and its halves); on a 16 KiB-a-lane
tpuv6e (tests/small_machine.py), with the plan's block choice made to take the biggest block (as an estimate that
undercounts would), the prefill's 512 x 512 and the decode's 2048-key step overflow the scratchpad, the Triton backend raises
OutOfResources, Inductor drops them and a smaller block runs, with the CPU's values. Breaks when either kernel is
launched with the one block the plan chose."""
import os

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
import small_machine  # noqa: E402

small_machine.use_machine(16)


def _run(q, k, v, causal):
    f = torch.compile(lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(
        a, b, c, is_causal=causal, enable_gqa=a.shape[1] != b.shape[1]), dynamic=False)
    got = f(q.to("npu:0"), k.to("npu:0"), v.to("npu:0")).float().cpu()
    rep = q.shape[1] // k.shape[1]
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1), is_causal=causal)
    return got, want


def _qkv(hq, h, l, s, d=128):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(1, n, t, d, generator=g).to(torch.bfloat16) for n, t in ((hq, l), (h, s), (h, s))]


@torch.no_grad()
def test_prefill_first_block_overflows():
    from pytorchsim.frontend.dispatcher import attention
    torch._dynamo.reset()
    q, k, v = _qkv(1, 1, 512, 512)
    with mock.patch.object(attention, "_choose_prefill_blocks", lambda L, S, *a: (512, 512)), \
            small_machine.Outcomes("BQ") as outcomes:
        got, want = _run(q, k, v, True)
    for cfg, what in outcomes:
        print(f"prefill candidate BQ={cfg['BQ']} BKV={cfg['BKV']}: {what}")
    assert outcomes[0][0]["BQ"] == 512 and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert any(w == "compiled" for _, w in outcomes), outcomes
    assert ((got - want).abs().max() / want.abs().max()).item() < 2e-2


@torch.no_grad()
def test_decode_first_step_overflows():
    from pytorchsim.frontend.dispatcher import attention
    torch._dynamo.reset()
    q, k, v = _qkv(8, 2, 1, 4096)
    with mock.patch.object(attention, "_choose_decode_step", lambda *a: 2048), \
            small_machine.Outcomes("BKV") as outcomes:
        got, want = _run(q, k, v, False)
    for cfg, what in outcomes:
        print(f"decode candidate BKV={cfg['BKV']}: {what}")
    assert outcomes[0][0]["BKV"] == 2048 and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert any(w == "compiled" for _, w in outcomes), outcomes
    assert ((got - want).abs().max() / want.abs().max()).item() < 2e-2


if __name__ == "__main__":
    test_prefill_first_block_overflows()
    test_decode_first_step_overflows()
    print("-----------------------------------\n|AttentionCandidateOverflow Test Passed|\n-----------------------------------")
