"""The paged decode kernel (TORCHSIM_KVCACHE_LAYOUT=tokenwise) is offered a config per candidate KV step; on a 8 KiB-a-lane
tpuv6e (tests/small_machine.py), with the plan's step choice made to take 2048 keys, that step overflows the scratchpad,
Inductor drops it and a smaller one runs, reading K and V in place, with the CPU's values. Breaks when the kernel is
launched with the one step the plan chose."""
import os

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["TORCHSIM_KVCACHE_LAYOUT"] = "tokenwise"

import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
import small_machine  # noqa: E402

small_machine.use_machine(8)


@torch.no_grad()
def test_paged_first_step_overflows(hq=8, h=2, s=4096, d=128):
    from one_layer import _paged
    from pytorchsim.frontend.dispatcher import attention
    g = torch.Generator().manual_seed(0)
    q, k, v = [torch.randn(1, n, t, d, generator=g).to(torch.bfloat16) for n, t in ((hq, 1), (h, s), (h, s))]
    kp, vp = _paged(k.to("npu:0"), v.to("npu:0"))
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float().repeat_interleave(hq // h, 1), v.float().repeat_interleave(hq // h, 1))
    torch._dynamo.reset()
    with mock.patch.object(attention, "_choose_decode_step", lambda *a: 2048), \
            small_machine.Outcomes("BKV") as outcomes:
        got = torch.compile(lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(
            a, b, c, enable_gqa=True), dynamic=False)(q.to("npu:0"), kp, vp).float().cpu()
    for cfg, what in outcomes:
        print(f"paged candidate BKV={cfg['BKV']}: {what}")
    assert outcomes[0][0]["BKV"] == 2048 and outcomes[0][1].startswith("OutOfResources"), outcomes
    assert any(w == "compiled" for _, w in outcomes), outcomes
    assert ((got - want).abs().max() / want.abs().max()).item() < 2e-2
    print("-----------------------------------\n|PagedAttentionCandidateOverflow Test Passed|\n-----------------------------------")


if __name__ == "__main__":
    test_paged_first_step_overflows()
