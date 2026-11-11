"""An int8 matmul widened to int32 (torchao's W8A8 on a device out_dtype has no _int_mm for) on the npu, against
the CPU. Cases: int_mm, int_mm_weight_t (a Linear's [N, K] weight), pass_graph (the mm is aten._int_mm, counted)
and pass_graph_int16 (an int16 one stays an mm); run with [case ...], all when none is named."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")


def _widened_mm(a, b):
    return torch.mm(a.to(torch.int32), b.to(torch.int32))


def _widened_mm_weight_t(a, w):
    return torch.mm(a.to(torch.int32), w.t().to(torch.int32))


#: name -> (fn, shapes)
CASES = {
    "int_mm": (_widened_mm, [(64, 512), (512, 256)]),
    "int_mm_weight_t": (_widened_mm_weight_t, [(64, 512), (256, 512)]),
}


def _inputs(shapes, dtype=torch.int8):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(-128, 128, s, generator=g).to(dtype) for s in shapes]


def run_case(name):
    fn, shapes = CASES[name]
    args = _inputs(shapes)
    ref = fn(*args)
    torch._dynamo.reset()
    out = torch.compile(fn, dynamic=False)(*[a.to(device) for a in args])
    check_result(f"IntMM {name}", out, ref)
    return True


def _post_grad(fn, args):
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    from pytorchsim.frontend.inductor import postgrad as pg
    gm = make_fx(fn, decomposition_table=select_decomp_table(), tracing_mode="fake")(*args)
    pg.CALLS.clear()
    pg.pytorchsim_post_grad_pass(gm)
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    return targets, pg.CALLS.get("int_mm")


def run_pass_graph(name):
    """On a traced graph: the widened int8 mm becomes aten._int_mm, counted."""
    targets, calls = _post_grad(_widened_mm_weight_t, _inputs([(64, 512), (256, 512)]))
    ok = (torch.ops.aten._int_mm.default in targets and torch.ops.aten.mm.default not in targets
          and calls == {"rewritten": 1})
    check_result("IntMM pass_graph", torch.tensor([ok]), torch.tensor([True]))
    return ok


def run_pass_graph_int16(name):
    """On a traced graph: an int16 mm widened to int32 is no _int_mm, and stays an mm."""
    targets, calls = _post_grad(_widened_mm, _inputs([(64, 512), (512, 256)], torch.int16))
    ok = (torch.ops.aten.mm.default in targets and torch.ops.aten._int_mm.default not in targets
          and calls is None)
    check_result("IntMM pass_graph_int16", torch.tensor([ok]), torch.tensor([True]))
    return ok


EXTRA = {"pass_graph": run_pass_graph, "pass_graph_int16": run_pass_graph_int16}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    for n in names:
        (EXTRA[n] if n in EXTRA else run_case)(n)
