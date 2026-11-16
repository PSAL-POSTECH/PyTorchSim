"""KNOWN FAILING (found while writing the megablox candidate test): the megablox grouped_mm on tpuv6e, K = N = 1024
over 3 groups of 100, 30 and 10 rows, stops in Spike on 'User store segfault' at 0x8069e000, in
triton_poi_fused_new_zeros_2 (the output's zero fill, not the megablox kernel); K = N = 256 and 512 run. It fails on the
tree before retiling too. Cause not found."""
import os

os.environ["TORCHSIM_GROUPED_MM_STRUCTURE"] = "megablox"

import torch


@torch.no_grad()
def test_megablox_k_n_1024(k=1024, n=1024, sizes=(100, 30, 10)):
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(sum(sizes), k, generator=g) * 0.1).to(torch.bfloat16)
    b = (torch.randn(len(sizes), k, n, generator=g) * 0.1).to(torch.bfloat16)
    offs = torch.cumsum(torch.tensor(sizes, dtype=torch.int32), 0, dtype=torch.int32)
    want = torch.cat([a[lo:hi].float() @ b[i].float() for i, (lo, hi) in enumerate(
        zip([0, *offs.tolist()[:-1]], offs.tolist()))])
    got = torch.compile(lambda *t: torch.nn.functional.grouped_mm(t[0], t[1], offs=t[2]), dynamic=False)(
        a.to("npu:0"), b.to("npu:0"), offs.to("npu:0")).float().cpu()
    assert ((got - want).abs().max() / want.abs().max()).item() < 2e-2


if __name__ == "__main__":
    test_megablox_k_n_1024()
