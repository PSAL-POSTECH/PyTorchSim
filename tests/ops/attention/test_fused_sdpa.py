"""SDPA on the npu through the overrideable backend: the fused prefill and decode kernels where
attention.fused_plan takes the call, SDPA's math where it does not, against a float32 CPU reference.

Each case states the route it must take (prefill, decode or fallback, SDPA's math) and checks
FUSED_SDPA.calls, the calls that ran each route: one launch per compiled call that takes a fused
kernel, one "fallback" count per math call run outside a compiled graph, nothing for math compiled
into the graph (Inductor's own kernels). So a gate that silently sends a fused case to math, or a
math case to a kernel, fails:

  prefill    causal GQA and non-causal, L == S; a head of 80 (Mllama's vision tower), a
             non-power-of-two S causal and not, and a non-causal L != S cross-attention block, each
             padded to the blocks a dot takes and masked back
  decode     GQA, L == 1, not causal; MLA's (DeepSeek-V3, Moonlight), k's D 192 = 128 + 64, v's 128;
             a head of 80 over a non-power-of-two S
  fallback   a causal decode (SDPA's is_causal is top-left: an L == 1 query attends key 0 alone),
             and a call whose inputs require grad (whose gradients are checked against the CPU's:
             the overrideable op has no backward)
  mask       a mask reaches the prefill kernel: Mllama's cross-attention mask as HF builds it (additive,
             [1, 1, L, S]) and as a bool, and one that masks a whole KV block out of every query row
             (the online softmax must take nothing from a block it masks whole); a mask on an L == 1
             decode stays the fallback
  shapes     plan() on meta tensors for the shapes the model suite brings, each a kernel or a
             Decline, so the gates are read without simulating: Mllama's vision tower (D 80, S 1601)
             and its cross-attention (L 32, S 6404), Qwen-VL's vision tower (D 80), and the
             top-left causal L != S that stays declined
  eager      a prefill-shaped call outside torch.compile runs math (no Triton launch outside a
             compiled graph) and says so in its calls; under inference_mode too, which skips the
             autograd key the op's npu kernel is registered at, eagerly and compiled
  spad       doubling a fused prefill's S keeps splash's block (one query block's scratch is live
             at a time); doubling its D halves the block as the scratchpad bound binds, and
             fused_plan declines only once one row per lane overruns the bound (checked on the plan,
             a Decline)
  counts     a compiled prefill called twice counts two launches, not its traces; under
             backend="eager" the call runs math, and only the fallback is counted
  layout     a v viewed and transposed out of a projection is read in place (one launch)
  attention  torchsim::attention's sliding window and logit softcap (gemma3's 512 window at S 1024, and
             at S 2048, where rows have no key of an earlier block in the window, gemma2's softcap) in the
             prefill and decode kernels, and an MLA prefill through SDPA (k's D
             192 = 128 + 64, v's 128), each against an explicit float32 reference

On a tree without FUSED_SDPA.calls the route checks are skipped and only the values are checked,
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

#  name: (route, B, Hq, H, L, S, D, causal[, v's D])
CASES = {
    "prefill_causal_gqa": ("_prefill_kernel", 1, 2, 1, 256, 256, 64, True),
    "prefill_noncausal": ("_prefill_kernel", 1, 2, 2, 256, 256, 64, False),
    "decode_gqa": ("_decode_kernel", 2, 8, 2, 1, 512, 64, False),
    "decode_mla": ("_decode_kernel", 1, 16, 16, 1, 512, 192, False, 128),
    "decode_causal_top_left": ("fallback", 1, 4, 1, 1, 128, 64, True),
    "prefill_non_pow2": ("_prefill_kernel", 1, 2, 2, 96, 96, 64, True),
    "prefill_head80": ("_prefill_kernel", 1, 2, 2, 128, 128, 80, False, 80),
    "prefill_pad_keys": ("_prefill_kernel", 1, 2, 2, 192, 192, 64, False),
    "prefill_cross": ("_prefill_kernel", 1, 4, 2, 32, 200, 64, False),
    "decode_head80_pad_keys": ("_decode_kernel", 1, 8, 2, 1, 300, 80, False, 80),
}

#  (B, Hq, H, L, S, D, DV, causal, the mask's shape or None): what the model suite brings, on meta tensors
SHAPES = {
    "mllama_vision": ((1, 16, 16, 1601, 1601, 80, 80, False, None), "_prefill_kernel"),
    "mllama_cross": ((1, 32, 8, 32, 6404, 128, 128, False, (1, 1, 32, 6404)), "_prefill_kernel"),
    "qwen_vl_vision": ((1, 16, 16, 16, 16, 80, 80, False, None), "_prefill_kernel"),
    "causal_l_not_s": ((1, 32, 8, 32, 6404, 128, 128, True, None), "Decline"),
    "mask_on_decode": ((1, 32, 8, 1, 6404, 128, 128, False, (1, 1, 1, 6404)), "Decline"),
    "mask_wrong_head": ((1, 32, 8, 32, 6404, 128, 128, False, (1, 4, 32, 6404)), "Decline"),
}

#  name: (route, B, Hq, H, L, S, D, the mask's kind); never causal, SDPA forbids a mask with is_causal
MASK_CASES = {
    "mask_cross_additive": ("_prefill_kernel", 1, 4, 2, 32, 200, 64, "additive"),
    "mask_cross_bool": ("_prefill_kernel", 1, 4, 2, 32, 200, 64, "bool"),
    "mask_blocks": ("_prefill_kernel", 1, 1, 1, 2048, 2048, 64, "blocks"),
    "mask_decode": ("fallback", 1, 8, 2, 1, 256, 64, "additive"),
}

TOL = 2e-2  # max |err| over max |ref|: bf16 inputs, f32 softmax state


def _inputs(B, Hq, H, L, S, D, dtype=torch.bfloat16, seed=0, DV=None):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Hq, L, D, generator=g).to(dtype),
            torch.randn(B, H, S, D, generator=g).to(dtype),
            torch.randn(B, H, S, DV or D, generator=g).to(dtype))


def _sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                          enable_gqa=q.shape[1] != k.shape[1])


def _err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _routed():
    from pytorchsim.frontend.dispatcher import attention
    return getattr(getattr(attention, "FUSED_SDPA", None), "calls", None)


def _route_ok(want):
    routed = _routed()
    return routed is None or dict(routed) == want


def _clear():
    if _routed() is not None:
        _routed().clear()


def run_case(name):
    route, B, Hq, H, L, S, D, causal, *dv = CASES[name]
    q, k, v = _inputs(B, Hq, H, L, S, D, DV=dv[0] if dv else None)
    ref = _sdpa(q.float(), k.float(), v.float(), causal)
    _clear()
    torch._dynamo.reset()
    f = torch.compile(lambda q, k, v: _sdpa(q, k, v, causal), dynamic=False)
    with torch.no_grad():
        out = f(q.to(device), k.to(device), v.to(device)).cpu()
    err = _err(out, ref)
    took = dict(_routed() or {})
    ok = err < TOL and _route_ok({} if route == "fallback" else {route: 1})
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
    compiled under inference_mode: the values match, and an eager call says it ran the fallback."""
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
    want = {"_prefill_kernel": 1} if compiled else {"fallback": 1}
    ok = err < TOL and _route_ok(want)
    print(f"{name}: route={dict(_routed() or {})} want={want} err={err:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_spad(name):
    """On meta tensors: doubling prefill_causal_gqa's S keeps splash's block, min(S, SPLASH_BLOCK),
    since one query block's scratch is live at a time; doubling its D halves the block as the
    scratchpad bound binds, until one row per lane overruns it and fused_plan declines."""
    from pytorchsim import config
    from pytorchsim.frontend.dispatcher import attention
    from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline
    _, B, Hq, H, _, S0, D0, causal = CASES["prefill_causal_gqa"]

    def plan(S, D):
        q, k, v = (torch.empty(B, h, S, D, dtype=torch.bfloat16, device="meta") for h in (Hq, H, H))
        return attention.fused_plan(q, k, v, causal)

    by_s = [(S, plan(S, D0)) for S in (S0 << i for i in range(13))]
    by_d = []
    D = D0
    while not by_d or not isinstance(by_d[-1][1], Decline):
        by_d.append((D, plan(attention.SPLASH_BLOCK, D)))
        D *= 2
    blocks = [p["BQ"] for _, p in by_d[:-1]]
    ok = (all(not isinstance(p, Decline) and p["BQ"] == p["BKV"] == min(S, attention.SPLASH_BLOCK) for S, p in by_s)
          and len(by_d) > 1 and blocks[0] == attention.SPLASH_BLOCK
          and all(a >= b for a, b in zip(blocks, blocks[1:]))
          and blocks[-1] == int(config.CONFIG_MACHINE.vector_lanes))
    print(f"{name}: S={by_s[-1][0]} {by_s[-1][1]}, D={by_d[-2][0]} {by_d[-2][1]}, D={by_d[-1][0]} declined "
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
    ok = max(errs) < TOL and (routed is None or took == [{"_prefill_kernel": 2}, {"fallback": 1}])
    print(f"{name}: inductor x2 route={took[0]} want={{'_prefill_kernel': 2}}, "
          f"eager route={took[1]} "
          f"want={{'fallback': 1}} err={max(errs):.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


#  name: (route, B, Hq, H, L, S, D, DV, window, softcap); D != DV goes through SDPA, the rest torchsim::attention
ATTENTION_CASES = {
    "window_prefill": ("_prefill_kernel", 1, 4, 1, 1024, 1024, 256, 256, 512, 0.0),
    "window_prefill_blocks": ("_prefill_kernel", 1, 2, 1, 2048, 2048, 256, 256, 512, 0.0),
    "window_prefill_blocks_d64": ("_prefill_kernel", 1, 1, 1, 2048, 2048, 64, 64, 512, 0.0),
    "window_prefill_whole_blocks": ("_prefill_kernel", 1, 1, 1, 4096, 4096, 64, 64, 3000, 0.0),
    "window_softcap_prefill": ("_prefill_kernel", 1, 2, 1, 1024, 1024, 64, 64, 384, 2.0),
    "window_softcap_decode": ("_decode_kernel", 1, 8, 2, 1, 1024, 64, 64, 384, 2.0),
    "mla_prefill": ("_prefill_kernel", 1, 2, 2, 256, 256, 192, 128, 0, 0.0),
}


def _banded(q, k, v, window, softcap):
    """Causal attention in float32 with the L queries the last L of the S keys, a key attended when it is
    less than `window` behind (0: any), the scores softcapped first."""
    L, S, group = q.shape[2], k.shape[2], q.shape[1] // k.shape[1]
    k, v = (t.float().repeat_interleave(group, dim=1) for t in (k, v))
    s = q.float() @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    behind = torch.arange(L)[:, None] + (S - L) - torch.arange(S)[None, :]
    s = s.masked_fill((behind < 0) | ((behind >= window) if window else False), float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def run_attention(name):
    """One ATTENTION_CASES case compiled on the npu: the route it must take and its values."""
    route, B, Hq, H, L, S, D, DV, window, softcap = ATTENTION_CASES[name]
    q, k, _ = _inputs(B, Hq, H, L, S, D)
    v = _inputs(B, Hq, H, L, S, DV, seed=1)[2]
    ref = _banded(q, k, v, window, softcap)
    if D != DV:
        call = lambda q, k, v: _sdpa(q, k, v, True)
    else:
        scale = D ** -0.5
        call = lambda q, k, v: torch.ops.torchsim.attention(q, k, v, scale, True, window, softcap)
    _clear()
    torch._dynamo.reset()
    f = torch.compile(call, dynamic=False)
    with torch.no_grad():
        out = f(q.to(device), k.to(device), v.to(device)).cpu()
    err = _err(out, ref)
    from pytorchsim.frontend.dispatcher import attention
    plan = attention.fused_plan(*(t.to("meta") for t in (q, k, v)), L > 1, window=window, softcap=softcap)
    ok = err < TOL and _route_ok({route: 1})
    print(f"{name}: route={dict(_routed() or {})} want={route} plan={plan} err={err:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_layout(name):
    """A prefill whose v is a projection's [B, S, H*D] viewed and transposed is read where it lies: one launch,
    no copy of v."""
    from pytorchsim.backend import launch_hooks
    B, Hq, H, S, D = 1, 2, 2, 256, 64
    q, k, v = _inputs(B, Hq, H, S, S, D)
    flat = v.transpose(1, 2).reshape(B, S, H * D).contiguous()
    ref = _banded(q, k, v, 0, 0.0)
    call = lambda q, k, flat: torch.ops.torchsim.attention(
        q, k, flat.view(B, S, H, D).transpose(1, 2), D ** -0.5, True, 0, 0.0)
    launches = []
    launch_hooks.observe_every_launch(lambda launcher: launches.append(getattr(launcher, "kernel_name", "?")))
    _clear()
    torch._dynamo.reset()
    with torch.no_grad():
        out = torch.compile(call, dynamic=False)(q.to(device), k.to(device), flat.to(device))
    err = _err(out.cpu(), ref)
    ok = err < TOL and len(launches) == 1
    print(f"{name}: launches={launches} err={err:.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_shapes(name):
    """Every SHAPES entry planned on meta tensors through FUSED_SDPA.plan, so the mask gate is read with
    the shape ones: the kernel it must take, or a Decline, without compiling or simulating anything."""
    from pytorchsim.frontend.dispatcher import attention
    from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline
    bad = []
    for case, ((B, Hq, H, L, S, D, DV, causal, mask), want) in SHAPES.items():
        q, k, v = (torch.empty(B, h, n, d, dtype=torch.bfloat16, device="meta")
                   for h, n, d in ((Hq, L, D), (H, S, D), (H, S, DV)))
        bias = None if mask is None else torch.empty(mask, dtype=torch.bfloat16, device="meta")
        plan = attention.FUSED_SDPA.plan(q, k, v, bias, 0.0, causal, D ** -0.5, 0, 0.0)
        got = "Decline" if isinstance(plan, Decline) else plan["kernel"].__name__
        print(f"  {case}: {got} {plan if got != 'Decline' else f'({plan})'}", flush=True)
        if got != want:
            bad.append(f"{case} took {got}, not {want}")
    print(f"{name}: {len(SHAPES) - len(bad)}/{len(SHAPES)} {'; '.join(bad) or ''} "
          f"{'PASS' if not bad else 'FAIL'}", flush=True)
    return not bad


def _mask_for(kind, L, S, dtype):
    """A [1, 1, L, S] mask: "blocks" gives each query row the half of the keys its own half is not in, so
    one whole KV block is masked out of every row; the others keep the first three quarters of the keys."""
    rows, cols = torch.arange(L)[:, None], torch.arange(S)[None, :]
    keep = (torch.where(rows < L // 2, cols >= S // 2, cols < S // 2) if kind == "blocks"
            else (cols < (S * 3) // 4).expand(L, S))
    keep = keep.reshape(1, 1, L, S).contiguous()
    if kind == "bool":
        return keep
    return torch.zeros(1, 1, L, S, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)


def run_masked(name):
    """One MASK_CASES case compiled on the npu: the route it must take with a mask, and its values against
    float32 SDPA given the same mask on the CPU."""
    route, B, Hq, H, L, S, D, kind = MASK_CASES[name]
    q, k, v = _inputs(B, Hq, H, L, S, D)
    mask = _mask_for(kind, L, S, torch.bfloat16)
    gqa = Hq != H
    call = lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=gqa)
    ref = call(q.float(), k.float(), v.float(), mask if mask.dtype == torch.bool else mask.float())
    _clear()
    torch._dynamo.reset()
    f = torch.compile(call, dynamic=False)
    with torch.no_grad():
        out = f(q.to(device), k.to(device), v.to(device), mask.to(device)).cpu()
    err = _err(out, ref)
    ok = err < TOL and _route_ok({} if route == "fallback" else {route: 1})
    print(f"{name}: route={dict(_routed() or {})} want={route} err={err:.3e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


EXTRA = {"shapes": run_shapes, "spad": run_spad, "requires_grad": run_grad, "eager": run_eager,
         "inference_mode": run_eager, "inference_mode_compiled": run_eager, "counts": run_counts, "layout": run_layout,
         **{n: run_attention for n in ATTENTION_CASES},
         **{n: run_masked for n in MASK_CASES}}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    results = [run_grad() if n == "requires_grad" else EXTRA[n](n) if n in EXTRA
               else run_case(n) for n in names]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
