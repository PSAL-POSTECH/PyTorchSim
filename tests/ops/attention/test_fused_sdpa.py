"""SDPA on the npu through the overrideable backend: the fused prefill and decode kernels where
attention.fused_plan takes the call, SDPA's math where it does not, against a float32 CPU reference.

Each case states the route it must take (prefill, decode or math) and checks attention.ROUTED, the
calls that ran each route: one launch per compiled call that takes a fused kernel, one count per
math call run outside a compiled graph, nothing for math compiled into the graph (Inductor's own
kernels). So a gate that silently sends a fused case to math, or a math case to a kernel, fails:

  prefill  causal GQA and non-causal, L == S
  decode   GQA, L == 1, not causal
  math     a causal decode (SDPA's is_causal is top-left: an L == 1 query attends key 0 alone),
           a non-power-of-two S, and a call whose inputs require grad (whose gradients are checked
           against the CPU's: the overrideable op has no backward)
  eager    a prefill-shaped call outside torch.compile runs math (no Triton launch outside a
           compiled graph) and says so in ROUTED; under inference_mode too, which skips the
           autograd key the op's npu kernel is registered at, eagerly and compiled
  spad     doubling a fused prefill's S, attention.fused_plan declines it at some S while every
           tile is still the same block, so only the scratchpad bound declines it (checked on
           the plan: a math run of that S on Spike takes far longer than this file)
  counts   a compiled prefill called twice counts two launches, not its traces; under
           backend="eager" the call runs math, and only math is counted

On a tree without attention.ROUTED the route checks are skipped and only the values are checked,
so the value failures a fix removes show on the tree before it.

Usage: test_fused_sdpa.py [case ...]   (all cases when none is named)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))

device = torch.device("npu:0")

#  name: (route, B, Hq, H, L, S, D, causal)
CASES = {
    "prefill_causal_gqa": ("prefill", 1, 2, 1, 256, 256, 64, True),
    "prefill_noncausal": ("prefill", 1, 2, 2, 256, 256, 64, False),
    "decode_gqa": ("decode", 2, 8, 2, 1, 512, 64, False),
    "decode_causal_top_left": ("math", 1, 4, 1, 1, 128, 64, True),
    "prefill_non_pow2": ("math", 1, 2, 2, 96, 96, 64, True),
}

TOL = 2e-2  # max |err| over max |ref|: bf16 inputs, f32 softmax state


def _inputs(B, Hq, H, L, S, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Hq, L, D, generator=g).to(dtype),
            torch.randn(B, H, S, D, generator=g).to(dtype),
            torch.randn(B, H, S, D, generator=g).to(dtype))


def _sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                          enable_gqa=q.shape[1] != k.shape[1])


def _err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _routed():
    from pytorchsim.frontend import attention
    return getattr(attention, "ROUTED", None)


def _route_ok(want):
    routed = _routed()
    return routed is None or dict(routed) == want


def _clear():
    if _routed() is not None:
        _routed().clear()


def run_case(name):
    route, B, Hq, H, L, S, D, causal = CASES[name]
    q, k, v = _inputs(B, Hq, H, L, S, D)
    ref = _sdpa(q.float(), k.float(), v.float(), causal)
    _clear()
    torch._dynamo.reset()
    f = torch.compile(lambda q, k, v: _sdpa(q, k, v, causal), dynamic=False)
    with torch.no_grad():
        out = f(q.to(device), k.to(device), v.to(device)).cpu()
    err = _err(out, ref)
    took = dict(_routed() or {})
    ok = err < TOL and _route_ok({} if route == "math" else {route: 1})
    print(f"{name}: route={took} want={route} err={err:.3e} {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def run_grad():
    """A call whose inputs require grad stays off the overrideable op, and its gradients match."""
    q, k, v = _inputs(1, 2, 2, 128, 128, 64, dtype=torch.float32)
    ref_in = [t.clone().requires_grad_() for t in (q, k, v)]
    _sdpa(*ref_in, True).sum().backward()
    dev_in = [t.to(device).requires_grad_() for t in (q, k, v)]
    _clear()
    try:
        out = _sdpa(*dev_in, True)
        out.sum().backward()
        errs = [_err(d.grad.cpu(), r.grad) for d, r in zip(dev_in, ref_in)]
    except Exception as exc:  # the backward of a route without one
        print(f"requires_grad: {type(exc).__name__}: {str(exc)[:200]} FAIL", flush=True)
        return False
    ok = _route_ok({}) and max(errs) < 1e-4
    print(f"requires_grad: route={dict(_routed() or {})} want={{}} grad err "
          f"{max(errs):.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_eager(name):
    """A prefill-shaped bf16 call outside torch.compile (under no_grad or inference_mode), or
    compiled under inference_mode: the values match, and an eager call says it ran math."""
    q, k, v = _inputs(1, 2, 2, 256, 256, 64)
    ref = _sdpa(q.float(), k.float(), v.float(), True)
    _clear()
    torch._dynamo.reset()
    compiled = name == "inference_mode_compiled"
    f = (torch.compile(lambda q, k, v: _sdpa(q, k, v, True), dynamic=False) if compiled
         else lambda q, k, v: _sdpa(q, k, v, True))
    ctx = torch.no_grad() if name == "eager" else torch.inference_mode()
    try:
        with ctx:
            out = f(q.to(device), k.to(device), v.to(device)).cpu()
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}: {str(exc)[:200]} FAIL", flush=True)
        return False
    err = _err(out, ref)
    want = {"prefill": 1} if compiled else {"math": 1}
    ok = err < TOL and _route_ok(want)
    print(f"{name}: route={dict(_routed() or {})} want={want} err={err:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_spad(name):
    """Double prefill_causal_gqa's S, on meta tensors, until fused_plan declines it: the block
    stays vector_lanes, so the tiles do not grow and the scratchpad bound is what declines."""
    from pytorchsim.frontend import attention
    _, B, Hq, H, _, S, D, causal = CASES["prefill_causal_gqa"]
    plans = []
    while S <= 1 << 20:
        q, k, v = (torch.empty(B, h, S, D, dtype=torch.bfloat16, device="meta")
                   for h in (Hq, H, H))
        plans.append((S, attention.fused_plan(q, k, v, causal)))
        if plans[-1][1] is None:
            break
        S *= 2
    declined = plans[-1][1] is None and len(plans) > 1
    ok = declined and all(p == plans[0][1] for _, p in plans[:-1])
    print(f"{name}: S={plans[0][0]} {plans[0][1]}, S={plans[-1][0]} {plans[-1][1]} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_counts(name):
    """A compiled prefill called twice records two launches, and the same call compiled with
    backend="eager" (Dynamo traces it, then runs it as eager ops) records the math it ran."""
    q, k, v = _inputs(1, 2, 2, 256, 256, 64)
    ref = _sdpa(q.float(), k.float(), v.float(), True)
    args = (q.to(device), k.to(device), v.to(device))
    took, errs = [], []
    for backend, calls in (("inductor", 2), ("eager", 1)):
        _clear()
        torch._dynamo.reset()
        f = torch.compile(lambda q, k, v: _sdpa(q, k, v, True), backend=backend, dynamic=False)
        with torch.no_grad():
            errs += [_err(f(*args).cpu(), ref) for _ in range(calls)]
        took.append(dict(_routed() or {}))
    routed = _routed()
    ok = max(errs) < TOL and (routed is None or took == [{"prefill": 2}, {"math": 1}])
    print(f"{name}: inductor x2 route={took[0]} want={{'prefill': 2}}, eager route={took[1]} "
          f"want={{'math': 1}} err={max(errs):.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


EXTRA = {"spad": run_spad, "requires_grad": run_grad, "eager": run_eager,
         "inference_mode": run_eager, "inference_mode_compiled": run_eager, "counts": run_counts}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    results = [run_grad() if n == "requires_grad" else EXTRA[n](n) if n in EXTRA
               else run_case(n) for n in names]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
