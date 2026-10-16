"""SDPA on the npu through the overrideable backend: the fused prefill and decode kernels where
attention.fused_plan takes the call, SDPA's math where it does not, against a float32 CPU reference.

Each case states the route it must take (prefill, decode or math) and checks the route the trace
took, so a gate that silently sends a fused case to math, or a math case to a kernel, fails here:

  prefill  causal GQA and non-causal, L == S
  decode   GQA, L == 1, not causal
  math     a causal decode (SDPA's is_causal is top-left: an L == 1 query attends key 0 alone),
           a non-power-of-two S, and a call whose inputs require grad (whose gradients are checked
           against the CPU's: the overrideable op has no backward)

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


def run_case(name):
    from pytorchsim.frontend import attention
    route, B, Hq, H, L, S, D, causal = CASES[name]
    q, k, v = _inputs(B, Hq, H, L, S, D)
    ref = _sdpa(q.float(), k.float(), v.float(), causal)
    attention.ROUTED.clear()
    torch._dynamo.reset()
    f = torch.compile(lambda q, k, v: _sdpa(q, k, v, causal), dynamic=False)
    with torch.no_grad():
        out = f(q.to(device), k.to(device), v.to(device)).cpu()
    err = _err(out, ref)
    took = dict(attention.ROUTED)
    ok = err < TOL and set(took) == {route}
    print(f"{name}: route={took} want={route} err={err:.3e} {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def run_grad():
    """A call whose inputs require grad stays off the overrideable op, and its gradients match."""
    from pytorchsim.frontend import attention
    q, k, v = _inputs(1, 2, 2, 128, 128, 64, dtype=torch.float32)
    ref_in = [t.clone().requires_grad_() for t in (q, k, v)]
    _sdpa(*ref_in, True).sum().backward()
    dev_in = [t.to(device).requires_grad_() for t in (q, k, v)]
    attention.ROUTED.clear()
    out = _sdpa(*dev_in, True)
    out.sum().backward()
    errs = [_err(d.grad.cpu(), r.grad) for d, r in zip(dev_in, ref_in)]
    ok = not attention.ROUTED and max(errs) < 1e-4
    print(f"requires_grad: route={dict(attention.ROUTED)} want={{}} grad err "
          f"{max(errs):.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, "requires_grad"]
    results = [run_grad() if n == "requires_grad" else run_case(n) for n in names]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
