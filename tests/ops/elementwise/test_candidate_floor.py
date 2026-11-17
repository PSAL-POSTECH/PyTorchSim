"""Llama 4's gather x sigmoid (128 x 5120 bf16) plus an i64 column holds over 4 bytes per element live: 524288, 262144
and 131072 overflow the scratchpad, so tiles go on down to a floor. Breaks if candidates stop at three
(NoTritonConfigsError); the unit check is red there, the gather alone fits at 65536 in this tree."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def test_candidates_reach_the_floor(device):
    from pytorchsim import config
    from pytorchsim.frontend import tiling

    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    shim = type("K", (), {"inside_reduction": False, "numels": {"xnumel": 655360}})()
    xs = [c["XBLOCK"] for c in tiling.candidate_blocks(shim, {"xnumel": 655360}, [{"dtype": "bfloat16"}])]
    print("candidates", xs)
    assert xs == sorted(xs, reverse=True) and len(set(xs)) == len(xs), xs
    assert xs[-1] <= lanes, xs
    sizes = tiling.candidate_sizes(1 << 16)
    assert sizes[-1] == 2, sizes


def test_gather_sigmoid(device, rows=128, hidden=5120):
    def fn(t1, t2, t3, t4, idx, gate, col):
        e = [torch.nn.functional.embedding(idx, t).float() for t in (t1, t2, t3, t4)]
        s = torch.sigmoid(gate[idx].float()).unsqueeze(-1)
        return (e[0] * s + e[1] * s * 2 + e[2] * s * 3 + e[3] * s * 4 + (idx.unsqueeze(-1) + col).float()).bfloat16()

    tables = [torch.randn(rows, hidden).to(torch.bfloat16).to(device) for _ in range(4)]
    idx = torch.randint(0, rows, (rows,)).to(device)
    gate = torch.randn(rows).to(torch.bfloat16).to(device)
    col = torch.arange(hidden).to(device)
    res = torch.compile(dynamic=False)(fn)(*tables, idx, gate, col)
    check_result("GatherSigmoid", res, fn(*[t.cpu() for t in tables], idx.cpu(), gate.cpu(), col.cpu()))


if __name__ == "__main__":
    test_candidates_reach_the_floor(torch.device("npu:0"))
    test_gather_sigmoid(torch.device("npu:0"))
