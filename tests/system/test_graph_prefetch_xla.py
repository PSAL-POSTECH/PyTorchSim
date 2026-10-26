"""The resident operand against XLA's pick on v6e (XLA 31eb6029) per AOT probe graph (run/llo_probe/xprog);
XLA-form attention runs with pattern_matcher off (Inductor fuses it into npu's SDPA, as Pallas), and scaled:
libtpu's flash_attention pass makes unscaled attention a custom call, not modelled (CI references: Pallas)."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
import torch.nn.functional as F  # noqa: E402
from pytorchsim.backend import codecache, compiler_bridge  # noqa: E402
from pytorchsim.frontend.inductor.autotune import graph_prefetch  # noqa: E402

BF = torch.bfloat16
QKV = [(1, 8, 1024, 128)] * 3


def _attn(q, k, v):
    """XLA's attention form: the scaled scores, an f32 softmax, the weighted values (xprog.py attn)."""
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    return torch.softmax(s.float(), dim=-1).to(BF) @ v


def _attn_causal(q, k, v):
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    keep = torch.tril(torch.ones(q.shape[2], q.shape[2], dtype=torch.bool, device=q.device))
    return torch.softmax(torch.where(keep, s.float(), -1e30), dim=-1).to(BF) @ v


def _flash(q, k, v):
    """The Pallas flash kernel's counterpart: SDPA, which npu runs as its fused attention kernel."""
    return F.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)


def _splash(q, k, v):
    """Splash as the v6e references call it: q pre-scaled, causal, through npu's fused attention kernel."""
    return F.scaled_dot_product_attention(q * q.shape[-1] ** -0.5, k, v, is_causal=True, scale=1.0)


def _mm(x, w):
    return x @ w


def _from_scores(s, v):
    """The probe's attention after its scores: scaled, an f32 softmax, the weighted values."""
    return torch.softmax((s * (v.shape[-1] ** -0.5)).float(), dim=-1).to(BF) @ v


def _attn_swap(q, k, v):
    """The scores written dot(k, q), then transposed."""
    return _from_scores((k @ q.transpose(-1, -2)).transpose(-1, -2), v)


def _attn_kt(q, kt, v):
    """k stored (B, H, D, S)."""
    return _from_scores(q @ kt, v)


W_SMALL, W_BIG = [(1024, 1024), (1024, 512)], [(1024, 1024), (1024, 4096)]
PROJ = [(1024, 1024), (1024, 1024), (1, 8, 1024, 128), (1, 8, 1024, 128)]
Q_BIG = [(1, 8, 2048, 128), (1, 8, 1024, 128), (1, 8, 1024, 128)]


def _attn_mm(q, k, v, x, w):
    return _attn(q, k, v), x @ w


def _flash_mm(q, k, v, x, w):
    return _flash(q, k, v), x @ w


def _splash_mm(q, k, v, x, w):
    return _splash(q, k, v), x @ w


def _proj_attn(x, wq, k, v):
    return _attn((x @ wq).reshape(1, 1024, 8, 128).transpose(1, 2), k, v)


#: name: (fn, input shapes, input names, XLA's pick or None, whether attention keeps XLA's form)
CASES = {
    "bmm_wbig": (_mm, [(8, 512, 1024), (8, 1024, 2048)], "xw", "w", False),
    "bmm_eq": (_mm, [(8, 1024, 1024), (8, 1024, 1024)], "xw", "x", False),
    "mm_bmm": (lambda x, w, a, b: (x @ w, a @ b),
               [(1024, 1024), (1024, 1024), (8, 2048, 1024), (8, 1024, 512)], "xwab", "a", False),
    "attn": (_attn, QKV, "qkv", "k", True),
    "attn_causal": (_attn_causal, QKV, "qkv", "k", True),
    "attn_mm_wsmall": (_attn_mm, QKV + W_SMALL, "qkvxw", "k", True),
    "attn_mm_wbig": (_attn_mm, QKV + W_BIG, "qkvxw", "w", True),
    "flash": (_flash, QKV, "qkv", None, False),
    "flash_mm_wsmall": (_flash_mm, QKV + W_SMALL, "qkvxw", "x", False),
    "splash_mm_wsmall": (_splash_mm, QKV + W_SMALL, "qkvxw", "x", False),
    "proj_attn": (_proj_attn, PROJ, ["x", "wq", "k", "v"], "k", True),
    "attn_qbig": (_attn, Q_BIG, "qkv", "q", True),
    "attn_vbig": (_attn, [(1, 8, 1024, 128), (1, 8, 1024, 128), (1, 8, 1024, 256)], "qkv", "v", True),
    "attn_qscaled_qbig": (lambda q, k, v: _attn(q * 0.5, k, v), Q_BIG, "qkv", "k", True),
    "mm_xscaled_xbig": (lambda x, w: (x * 2) @ w, [(4096, 1024), (1024, 512)], "xw", "w", False),
    "mm_xexp_xbig": (lambda x, w: torch.exp(x) @ w, [(4096, 1024), (1024, 512)], "xw", "w", False),
    "mm_xalsoadd": (lambda x, w, y: (x @ w, x + y), [(4096, 1024), (1024, 512), (4096, 1024)], "xwy", "w",
                    False),
    "add_only": (lambda x, y: x + y, [(4096, 1024), (4096, 1024)], "xy", None, False),
    "reduce_only": (lambda x: torch.sum(x, dim=1), [(4096, 1024)], "x", "x", False),
    "linear_wT_wbig": (lambda x, w: x @ w.T, [(512, 1024), (4096, 1024)], "xw", "w", False),
    "addmm_bias_big": (lambda b, x, w: b + x @ w, [(4096, 1024), (4096, 256), (256, 1024)], "bxw", "x",
                       False),
    "addmm_bias_row": (lambda b, x, w: b + x @ w, [(1, 8192), (128, 256), (256, 8192)], "bxw", "w", False),
    "attn_swap": (_attn_swap, QKV, "qkv", "q", True),
    "attn_kT": (_attn_kt, [(1, 8, 1024, 128), (1, 8, 128, 1024), (1, 8, 1024, 128)], ["q", "kt", "v"], "kt",
                True),
    "mm4d": (_mm, [(1, 8, 1024, 1024)] * 2, "xw", "w", False),
    "bmm_x4": (lambda x, w: x.reshape(8, 1024, 1024) @ w, [(1, 8, 1024, 1024), (8, 1024, 1024)], "xw", "w",
               False),
    "bmm_w4": (lambda x, w: x @ w.reshape(8, 1024, 1024), [(8, 1024, 1024), (1, 8, 1024, 1024)], "xw", "x",
               False),
    "mm_wT": (lambda x, w: x @ w.T, [(1024, 1024)] * 2, "xw", "x", False),
    "mm_xT": (lambda x, w: x.T @ w, [(1024, 1024)] * 2, "xw", "x", False),
    "mm_xTwT": (lambda x, w: x.T @ w.T, [(1024, 1024)] * 2, "xw", "x", False),
    "bmm_wT": (lambda x, w: x @ w.transpose(1, 2), [(8, 1024, 1024)] * 2, "xw", "x", False),
    "bmm_xT": (lambda x, w: x.transpose(1, 2) @ w, [(8, 1024, 1024)] * 2, "xw", "x", False),
    "mm_xclone_xbig": (lambda x, w: x.clone() @ w, [(4096, 1024), (1024, 512)], "xw", "x", False),
    "mm_xclone_xbig_also": (lambda x, w: (x.clone() @ w, x.clone()), [(4096, 1024), (1024, 512)], "xw", "w",
                            False),
}
BUDGET_CASES = {
    "probe_w96m": (_mm, [(256, 4096), (4096, 12288)], "xw", "w", False),
    "probe_w112m": (_mm, [(256, 4096), (4096, 14336)], "xw", "x", False),
}


def pick(fn, shapes, names, xla_form):
    """The name of the input the graph holds once Inductor lowers `fn` over bf16 inputs of `shapes` (the
    compiler faked away), None for none; `xla_form` keeps attention as XLA writes it (no Inductor fusion)."""
    from torch._inductor import config as inductor_config
    got, dynamo = {}, []

    def spy(graph, real=graph_prefetch.elected):
        out = real(graph)
        ph = [n.target for n in graph.module.graph.find_nodes(op="placeholder")]
        got["held"] = [dynamo[ph.index(t)] for t in out]
        return out

    def backend(gm, example):
        from torch._inductor.compile_fx import compile_fx
        ptrs = [x.data_ptr() for x in xs]
        dynamo[:] = [names[ptrs.index(x.data_ptr())] for x in example]
        return compile_fx(gm, example)
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(*s, generator=g).to(BF).to("npu:0") for s in shapes]
    torch._dynamo.reset()
    with mock.patch.object(codecache, "torchsim_compile", lambda src, meta, kernel_name: lambda *a: None), \
            mock.patch.object(graph_prefetch, "elected", spy), \
            inductor_config.patch(fx_graph_cache=False, pattern_matcher=not xla_form):
        torch.compile(fn, dynamic=False, backend=backend)(*xs)
    held = got.get("held") or [None]
    return held[0]


def test_every_case_picks_what_xla_picks():
    """Breaks if any row's elected input differs from the one XLA cross-program prefetched."""
    wrong = {}
    for name, (fn, shapes, names, xla, xla_form) in CASES.items():
        ours = pick(fn, shapes, names, xla_form)
        print(f"{name}: XLA {xla}, ours {ours}")
        if ours != xla:
            wrong[name] = (xla, ours)
    assert not wrong, wrong


def test_the_budget_rows_pick_what_xla_picks():
    """Breaks if a held weight at XLA's VMEM budget (96 MiB on v6e, VMEM less scoped VMEM) is not held, or one
    past it is; xfail until the machine states the scoped VMEM (compiler: spad_scoped_kb_per_lane: 256)."""
    cn, target = compiler_bridge.compiler_contract("copy_node"), compiler_bridge.target()
    if cn.resident_budget(target) == target.spad_size:
        raise unittest.SkipTest("xfail: the machine states no scoped VMEM (compiler.spad_scoped_kb_per_lane), so "
                                "the resident budget is the whole bank, not XLA's 96 MiB")
    wrong = {n: (c[3], pick(c[0], c[1], c[2], c[4])) for n, c in BUDGET_CASES.items()}
    assert all(xla == ours for xla, ours in wrong.values()), wrong


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except unittest.SkipTest as e:
                print(f"SKIP {name}: {e}")
                continue
            print(f"PASS {name}")
