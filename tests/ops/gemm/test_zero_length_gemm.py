"""A matmul with a zero-length M, N, or K (unbiased) on the npu is zeros of its output shape, against the CPU;
an addmm with K = 0 is its bias, left to Inductor. Cases: mm_m0, mm_n0, mm_k0, bmm_n0, bmm_k0, addmm_m0, and
pass_graph (no zero-length mm left, and counted); run with [case ...], all when none is named."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")

#: name -> (op, shapes)
CASES = {
    "mm_m0": (torch.mm, [(0, 8), (8, 4)]),
    "mm_n0": (torch.mm, [(4, 8), (8, 0)]),
    "mm_k0": (torch.mm, [(4, 0), (0, 8)]),
    "bmm_n0": (torch.bmm, [(2, 4, 8), (2, 8, 0)]),
    "bmm_k0": (torch.bmm, [(2, 4, 0), (2, 0, 8)]),
    "addmm_m0": (torch.addmm, [(4,), (0, 8), (8, 4)]),
}


def _inputs(shapes):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(*s, generator=g) for s in shapes]


def run_case(name):
    op, shapes = CASES[name]
    args = _inputs(shapes)
    ref = op(*args)
    torch._dynamo.reset()
    out = torch.compile(op, dynamic=False)(*[a.to(device) for a in args])
    check_result(f"ZeroLengthGemm {name}", out, ref)
    return True


def run_pass_graph(name):
    """On a traced graph: an mm with K = 0 becomes zeros, counted."""
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    from pytorchsim.frontend.inductor import postgrad as pg
    a, b = _inputs([(4, 0), (0, 8)])
    gm = make_fx(torch.mm, decomposition_table=select_decomp_table(), tracing_mode="fake")(a, b)
    pg.CALLS.clear()
    pg.pytorchsim_post_grad_pass(gm)
    left = gm.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
    ok = not left and pg.CALLS.get("gemm_zero_length") == {"rewritten": 1}
    check_result("ZeroLengthGemm pass_graph", torch.tensor([ok]), torch.tensor([True]))
    return ok


EXTRA = {"pass_graph": run_pass_graph}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    for n in names:
        (EXTRA[n] if n in EXTRA else run_case)(n)
