"""Eager GQA attention as transformers writes it (repeat_kv: unsqueeze, expand, reshape) takes the shared key
and value heads, not repeat_kv's copies: with a mask both bmms (gqa_shared_kv), without one the fused SDPA
it becomes (gqa_shared_kv_fused); the output matches the CPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def repeat_kv(x, n_rep):
    """transformers' repeat_kv: [B, H, S, D] -> [B, H * n_rep, S, D]."""
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def attention(q, k, v, mask=None):
    """transformers' eager_attention_forward without dropout: the mask added to the scores."""
    n_rep = q.shape[1] // k.shape[1]
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    w = torch.matmul(q, k.transpose(2, 3)) * (q.shape[-1] ** -0.5)
    if mask is not None:
        w = w + mask
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, v)


def run(name, rewrite, batch, heads, kv_heads, q_len, kv_len, head_dim, dtype=torch.float32, masked=True):
    """Compile `attention` on npu for one shape, compare it with the CPU, and check `rewrite` took it."""
    from pytorchsim.frontend.inductor import postgrad
    torch._dynamo.reset()
    for r in ("gqa_shared_kv", "gqa_shared_kv_fused"):
        postgrad.CALLS.pop(r, None)
    torch.manual_seed(0)
    q = torch.randn(batch, heads, q_len, head_dim, dtype=dtype)
    k = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype)
    v = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype)
    mask = torch.triu(torch.full((q_len, kv_len), -1e4, dtype=dtype), kv_len - q_len + 1) if masked else None
    want = attention(q, k, v, mask)
    dev = [t.to("npu:0") if t is not None else None for t in (q, k, v, mask)]
    got = torch.compile(dynamic=False)(attention)(*dev)
    tol = (2e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    check_result(name, got.cpu(), want, *tol)
    took = postgrad.CALLS.get(rewrite, {}).get("rewritten", 0)
    expect = 2 if rewrite == "gqa_shared_kv" else 1
    assert took == expect, f"{name}: {rewrite} took {took}, not {expect}"
    print(f"PASS  {name}: {rewrite} x{took}")


if __name__ == "__main__":
    run("decode", "gqa_shared_kv", batch=2, heads=8, kv_heads=2, q_len=1, kv_len=64, head_dim=32)
    run("prefill", "gqa_shared_kv", batch=2, heads=8, kv_heads=2, q_len=16, kv_len=16, head_dim=32)
    run("square", "gqa_shared_kv", batch=1, heads=4, kv_heads=2, q_len=8, kv_len=32, head_dim=32)
    run("bf16 decode", "gqa_shared_kv", batch=2, heads=8, kv_heads=2, q_len=1, kv_len=64, head_dim=32,
        dtype=torch.bfloat16)
    run("fused decode", "gqa_shared_kv_fused", batch=2, heads=8, kv_heads=2, q_len=1, kv_len=64, head_dim=32,
        masked=False)
    print("Test Passed")
