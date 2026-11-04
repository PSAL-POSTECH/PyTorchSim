"""A causal prefill without a window (torchsim::attention, q 16 heads, k and v 8, S 4096, D 128) in timing mode is
under 800,000 TOGSim cycles: 753,365 with each query block's earlier KV blocks taken before its diagonal one,
889,355 with the diagonal first (d91d088's order, meant for a window only)."""
import logging
import os
import sys

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)
os.environ["TORCHSIM_TIMING_MODE"] = "1"
os.environ.setdefault("TORCHSIM_FUNCTIONAL_MODE", "0")

import re  # noqa: E402
import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim.backend import codecache  # noqa: E402,F401  its logger set up before a handler joins it

HQ, HKV, S, D = 16, 8, 4096, 128
BOUND = 800_000


class _Cycles(logging.Handler):
    def __init__(self):
        super().__init__()
        self.got = {}

    def emit(self, record):
        m = re.search(r"\[TOGSim\] (\S+) simulated: (\d+) cycles", record.getMessage())
        if m:
            self.got[m.group(1)] = int(m.group(2))


def test_a_causal_prefill_takes_its_earlier_blocks_before_the_diagonal():
    seen = _Cycles()
    logging.getLogger("pytorchsim.backend.codecache").addHandler(seen)
    dev = torch.device("npu:0")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, h, S, D, generator=g).bfloat16().to(dev) for h in (HQ, HKV, HKV))
    with torch.no_grad():
        torch.compile(lambda q, k, v: torch.ops.torchsim.attention(q, k, v, D ** -0.5, True, 0, 0.0))(q, k, v)
    cycles = [c for name, c in seen.got.items() if "prefill_kernel" in name]
    assert len(cycles) == 1, seen.got
    assert cycles[0] <= BOUND, f"{cycles[0]} cycles, over {BOUND}"
    return cycles[0]


if __name__ == "__main__":
    print(f"cycles {test_a_causal_prefill_takes_its_earlier_blocks_before_the_diagonal()}")
    print("Attention Causal Cycles Test Passed")
