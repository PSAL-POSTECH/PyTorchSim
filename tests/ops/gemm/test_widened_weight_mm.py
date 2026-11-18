"""A weight-only int8 matmul (W8A16: an int8 weight widened to bf16, then a bf16 mm) on the npu, against the
same math in float32 on the CPU. Cases: weight_t (a Linear's [N, K] weight, transposed, as torchao's), weight_kn
(stored [K, N]), row (a 1 x K input) and pass_compiled (compiled, the transposed widening is stored row-major,
counted once; the row-major one is left); run with [case ...], all when none is named."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")


def _weight_t(x, w):
    return torch.mm(x, w.t().to(torch.bfloat16))


def _weight_kn(x, w):
    return torch.mm(x, w.to(torch.bfloat16))


#: name -> (fn, x shape, w shape, whether w is stored [N, K])
CASES = {
    "weight_t": (_weight_t, (64, 512), (256, 512), True),
    "weight_kn": (_weight_kn, (64, 512), (512, 256), False),
    "row": (_weight_t, (1, 512), (256, 512), True),
}


def _inputs(x_shape, w_shape):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(x_shape, generator=g).to(torch.bfloat16)
    w = torch.randint(-128, 128, w_shape, generator=g).to(torch.int8)
    return x, w


def run_case(name):
    """Compile the case on the npu; the reference is the same widened weight and input in float32."""
    fn, xs, ws, nk = CASES[name]
    x, w = _inputs(xs, ws)
    ref = torch.mm(x.float(), w.float().t() if nk else w.float()).to(torch.bfloat16).float()
    torch._dynamo.reset()
    out = torch.compile(fn, dynamic=False)(x.to(device), w.to(device)).cpu().float()
    check_result(f"WidenedWeightMM {name}", out, ref, rtol=1e-2, atol=ref.abs().max().item() * 1e-2)
    return True


def run_pass_compiled(name):
    """Compiled on the npu with Inductor's caches off (so the pass runs): the transposed widening is stored
    row-major by mm_widened_weight, once; the row-major one is left alone."""
    from torch._inductor import config as inductor_config

    from pytorchsim.frontend.inductor import postgrad as pg
    counts = []
    for fn, xs, ws, _ in (CASES["weight_t"], CASES["weight_kn"]):
        x, w = _inputs(xs, ws)
        pg.CALLS.clear()
        torch._dynamo.reset()
        with inductor_config.patch(force_disable_caches=True):
            torch.compile(fn, dynamic=False)(x.to(device), w.to(device))
        counts.append(pg.CALLS.get("mm_widened_weight"))
    ok = counts == [{"rewritten": 1}, None]
    print("mm_widened_weight counts (weight_t, weight_kn):", counts)
    check_result("WidenedWeightMM pass_compiled", torch.tensor([ok]), torch.tensor([True]))
    return ok


EXTRA = {"pass_compiled": run_pass_compiled}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    for n in names:
        (EXTRA[n] if n in EXTRA else run_case)(n)
