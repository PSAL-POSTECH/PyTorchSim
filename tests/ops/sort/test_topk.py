"""torch.topk on the npu: the topk kernel where TopK.plan takes the call, the reference rounds as
aten ops where it does not, against torch.topk on the CPU.

Each case states the route it must take and checks topk.ROUTED: one "kernel" count per launch of a
compiled call the kernel takes, one "reference" count per eager call, nothing for a reference
compiled into the graph (Inductor's own kernels). Ties take the first index, as a stable
descending sort does.

  kernel     rows a whole number of programs and not, a row length that is not a power of two,
             the smallest k, bf16, ties
  reference  topk along a dim that is not the last
  grad       a compiled call whose input requires grad runs the kernel, and its gradient is aten's
  eager      outside torch.compile, and under inference_mode eagerly and compiled
  counts     a compiled call run twice counts two launches, not its traces

Usage: test_topk.py [case ...]   (all cases when none is named)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))

device = torch.device("npu:0")

#  name: (route, shape, k, dim, largest, dtype)
CASES = {
    "router": ("kernel", (64, 128), 8, -1, True, torch.float32),
    "ragged_rows": ("kernel", (100, 128), 5, -1, True, torch.float32),
    "row_not_pow2": ("kernel", (32, 100), 3, -1, True, torch.float32),
    "smallest": ("kernel", (64, 128), 4, -1, False, torch.float32),
    "bf16": ("kernel", (64, 128), 8, -1, True, torch.bfloat16),
    "ties": ("kernel", (64, 128), 8, -1, True, torch.float32),
    "dim0": ("reference", (128, 32), 4, 0, True, torch.float32),
}


def _input(shape, dtype, ties=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    return ((x * 4).round() if ties else x).to(dtype)


def _ref(x, k, dim, largest):
    """Values and indices of a stable sort: among equal values the first index comes first."""
    order = torch.sort(x.float(), dim=dim, descending=largest, stable=True).indices.narrow(dim, 0, k)
    return x.gather(dim, order), order


def _routed():
    from pytorchsim.frontend.dispatcher import topk
    return topk.ROUTED


def _check(name, got, want, route, took):
    """Values and indices equal, and ROUTED is `route` (a dict of counts)."""
    ok = torch.equal(got[0].cpu(), want[0]) and torch.equal(got[1].cpu(), want[1]) and took == route
    print(f"{name}: route={took} want={route} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_case(name):
    route, shape, k, dim, largest, dtype = CASES[name]
    x = _input(shape, dtype, ties=name == "ties")
    _routed().clear()
    torch._dynamo.reset()
    f = torch.compile(lambda t: torch.topk(t, k, dim=dim, largest=largest), dynamic=False)
    want = {} if route == "reference" else {route: 1}
    return _check(name, f(x.to(device)), _ref(x, k, dim, largest), want, dict(_routed()))


def run_grad(name):
    """values.sum().backward() through the compiled kernel: the gradient is 1 at each index taken."""
    x = _input((64, 128), torch.float32)
    ref = x.clone().requires_grad_()
    torch.topk(ref, 8, -1).values.sum().backward()
    dev = x.to(device).requires_grad_()
    _routed().clear()
    torch._dynamo.reset()
    v, i = torch.compile(lambda t: torch.topk(t, 8, -1), dynamic=False)(dev)
    v.sum().backward()
    ok = torch.equal(dev.grad.cpu(), ref.grad) and dict(_routed()) == {"kernel": 1}
    print(f"{name}: route={dict(_routed())} want=kernel {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_eager(name):
    """Outside torch.compile the reference runs (one count); compiled under inference_mode, the
    kernel."""
    x = _input((64, 128), torch.float32)
    want = _ref(x, 8, -1, True)
    _routed().clear()
    torch._dynamo.reset()
    compiled = name == "inference_mode_compiled"
    f = (torch.compile(lambda t: torch.topk(t, 8, -1), dynamic=False) if compiled
         else lambda t: torch.topk(t, 8, -1))
    ctx = torch.no_grad() if name == "eager" else torch.inference_mode()
    with ctx:
        got = f(x.to(device))
    return _check(name, got, want, {"kernel": 1} if compiled else {"reference": 1}, dict(_routed()))


def run_counts(name):
    """A compiled call run twice counts two launches."""
    x = _input((64, 128), torch.float32)
    _routed().clear()
    torch._dynamo.reset()
    f = torch.compile(lambda t: torch.topk(t, 8, -1), dynamic=False)
    for _ in range(2):
        f(x.to(device))
    ok = dict(_routed()) == {"kernel": 2}
    print(f"{name}: route={dict(_routed())} want={{'kernel': 2}} {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


EXTRA = {"grad": run_grad, "eager": run_eager, "inference_mode": run_eager,
         "inference_mode_compiled": run_eager, "counts": run_counts}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    results = [EXTRA[n](n) if n in EXTRA else run_case(n) for n in names]
    if not all(results):
        sys.exit(1)
    print("PASS")
