"""A model's SDPA call over repeat_kv's copies (transformers' sdpa attention with a mask) hands the fused kernel the
shared K and V heads: the pre-grad pass rewrites it to enable_gqa. Breaks when the kernel is planned over the
H*R copies again (Qwen3-8B decode at 1024: 16 MiB of K and V read, not 4). Values match the CPU."""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result  # noqa: E402


def repeat_kv(x, n_rep):
    """transformers' repeat_kv: [B, H, S, D] -> [B, H * n_rep, S, D]."""
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def call(q, k, v, m):
    """transformers' sdpa_attention_forward with a mask: repeat_kv, then SDPA without enable_gqa."""
    n_rep = q.shape[1] // k.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                                            attn_mask=m)


def run(name, batch, heads, kv_heads, q_len, kv_len, head_dim, dtype=torch.bfloat16):
    """Compile `call` on npu for one shape: the last plan, AOT's, the kernel takes, is over kv_heads key heads
    (Dynamo's fake run plans before the pre-grad pass, over the copies); values match."""
    from pytorchsim.frontend.dispatcher import attention
    torch._dynamo.reset()
    planned = []
    plan = attention.FUSED_SDPA.plan
    attention.FUSED_SDPA.plan = lambda q, k, *a: (planned.append(k.shape[1]), plan(q, k, *a))[1]
    try:
        torch.manual_seed(0)
        q = torch.randn(batch, heads, q_len, head_dim, dtype=dtype)
        k = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype)
        v = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype)
        mask = torch.triu(torch.full((q_len, kv_len), -1e4, dtype=dtype), kv_len - q_len + 1)[None, None]
        want = call(q.float(), k.float(), v.float(), mask.float())
        got = torch.compile(dynamic=False)(call)(*[t.to("npu:0") for t in (q, k, v, mask)])
    finally:
        attention.FUSED_SDPA.plan = plan
    check_result(name, got.cpu(), want, 2e-2, 1e-2)
    assert planned and planned[-1] == kv_heads, f"{name}: the kernel planned over {planned[-1:]} key heads " \
        f"(every plan {planned}), not the {kv_heads} shared ones"
    print(f"PASS  {name}: planned over {kv_heads} key heads for {heads} query heads")


if __name__ == "__main__":
    run("decode", batch=1, heads=8, kv_heads=2, q_len=1, kv_len=256, head_dim=64)
    run("prefill", batch=2, heads=8, kv_heads=2, q_len=16, kv_len=16, head_dim=32)
    print("Test Passed")
