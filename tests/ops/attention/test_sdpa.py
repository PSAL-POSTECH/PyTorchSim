import sys
import os
import torch
import torch._dynamo
import torch.nn.functional as F

base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.append(base_dir)

device = torch.device("npu:0")

# ---------------------------------------------------------------------------
# Unified SDPA test: Flash attention (MHA) and GQA, each in prefill and decode,
# swept over several dimensions and compared against the CPU MATH backend
# (numerical ground truth). 
#
#   - prefill : query length L == S (full sequence)
#   - decode  : query length L == 1 (single new token over a KV cache of length S)
#   - MHA     : query heads == kv heads          (enable_gqa=False)
#   - GQA     : query heads == ratio * kv heads  (enable_gqa=True)
# ---------------------------------------------------------------------------
BATCH_LIST        = [1, 2, 4]      # batch size
HEAD_LIST         = [1, 4, 8]      # MHA: query == kv heads
SEQ_LIST          = [128, 256, 512]     # KV sequence length S (multiples of tile_s)
HEAD_DIM_LIST     = [64, 128]      # head dim D (requires e == ev)

# GQA head layout as (num_kv_heads, group_ratio); Hq = num_kv_heads * ratio.
GQA_HEAD_CONFIGS  = [
    (1, 2), (1, 4), (1, 5), (1, 8), (1, 16),
    (2, 8), (4, 3), (4, 8), (8, 8),
]
GQA_SEQ_LIST      = [128, 256, 512]
GQA_HEAD_DIM_LIST = [64, 128]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clear_caches():
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch._inductor.codecache import FxGraphCache
    AOTAutogradCache.clear()
    torch._dynamo.reset()
    os.environ["TORCHINDUCTOR_CACHE"] = "0"
    FxGraphCache.clear()


_RESULTS = {"pass": 0, "fail": 0, "failed_names": []}


def assert_close(name, out, cpu_out, rtol=1e-2, atol=1e-2):
    """Compare device output vs CPU MATH reference. Records the result and keeps
    going (does not abort) so the whole sweep runs; the summary/exit code is
    emitted at the end of __main__."""
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print(f"[PASS] {name}")
        _RESULTS["pass"] += 1
        return True
    print(f"[FAIL] {name}")
    print("  device out:", out.cpu())
    print("  cpu    out:", cpu_out)
    _RESULTS["fail"] += 1
    _RESULTS["failed_names"].append(name)
    return False


def _run_sdpa(device, q, k, v, **kwargs):
    """Compile and run SDPA on device; return result on device."""
    opt_fn = torch.compile(dynamic=False)(F.scaled_dot_product_attention)
    return opt_fn(q.to(device), k.to(device), v.to(device), **kwargs)


def _cpu_sdpa(q, k, v, **kwargs):
    """Reference SDPA on CPU. Force the MATH backend (numerical ground truth):
    it supports GQA and is independent of any outer sdpa_kernel() context, so it
    works even when the device run is wrapped in a flash-only context."""
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), **kwargs)


def _check(name, q, k, v, **kwargs):
    """Compile+run on device, compare against CPU MATH, record result."""
    clear_caches()
    out     = _run_sdpa(device, q, k, v, **kwargs)
    cpu_out = _cpu_sdpa(q, k, v, **kwargs)
    assert_close(name, out, cpu_out)


# ---------------------------------------------------------------------------
# Flash attention (MHA: query heads == kv heads)
# ---------------------------------------------------------------------------
def test_flash_prefill(device):
    """MHA prefill: q/k/v = (B, H, S, D), query length L == S."""
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False)
    for B in BATCH_LIST:
        for H in HEAD_LIST:
            for S in SEQ_LIST:
                for D in HEAD_DIM_LIST:
                    q = torch.rand(B, H, S, D, dtype=torch.float16)
                    k = torch.rand(B, H, S, D, dtype=torch.float16)
                    v = torch.rand(B, H, S, D, dtype=torch.float16)
                    _check(f"flash-prefill(B{B},H{H},S{S},D{D})", q, k, v, **kwargs)


def test_flash_decode(device):
    """MHA decode: q = (B, H, 1, D), k/v = (B, H, S, D), query length L == 1."""
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False)
    for B in BATCH_LIST:
        for H in HEAD_LIST:
            for S in SEQ_LIST:
                for D in HEAD_DIM_LIST:
                    q = torch.rand(B, H, 1, D, dtype=torch.float16)
                    k = torch.rand(B, H, S, D, dtype=torch.float16)
                    v = torch.rand(B, H, S, D, dtype=torch.float16)
                    _check(f"flash-decode(B{B},H{H},S{S},D{D},L1)", q, k, v, **kwargs)


# ---------------------------------------------------------------------------
# GQA (query heads = ratio * kv heads)
# ---------------------------------------------------------------------------
def test_gqa_prefill(device):
    """GQA prefill: q = (B, Hq, S, D), k/v = (B, Hkv, S, D), Hq = ratio*Hkv, L == S."""
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True)
    for B in BATCH_LIST:
        for Hkv, ratio in GQA_HEAD_CONFIGS:
            Hq = ratio * Hkv
            for S in GQA_SEQ_LIST:
                for D in GQA_HEAD_DIM_LIST:
                    q = torch.rand(B, Hq,  S, D, dtype=torch.float16)
                    k = torch.rand(B, Hkv, S, D, dtype=torch.float16)
                    v = torch.rand(B, Hkv, S, D, dtype=torch.float16)
                    _check(f"gqa-prefill(B{B},Hq{Hq},Hkv{Hkv},S{S},D{D})", q, k, v, **kwargs)


def test_gqa_decode(device):
    """GQA decode: q = (B, Hq, 1, D), k/v = (B, Hkv, S, D), Hq = ratio*Hkv, L == 1."""
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True)
    for B in BATCH_LIST:
        for Hkv, ratio in GQA_HEAD_CONFIGS:
            Hq = ratio * Hkv
            for S in GQA_SEQ_LIST:
                for D in GQA_HEAD_DIM_LIST:
                    q = torch.rand(B, Hq,  1, D, dtype=torch.float16)
                    k = torch.rand(B, Hkv, S, D, dtype=torch.float16)
                    v = torch.rand(B, Hkv, S, D, dtype=torch.float16)
                    _check(f"gqa-decode(B{B},Hq{Hq},Hkv{Hkv},S{S},D{D},L1)", q, k, v, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        test_flash_prefill(device)
        test_flash_decode(device)
        test_gqa_prefill(device)
        test_gqa_decode(device)
    
    total = _RESULTS["pass"] + _RESULTS["fail"]
    print("=" * 60)
    print(f"SDPA tests: {_RESULTS['pass']}/{total} passed, {_RESULTS['fail']} failed")
    if _RESULTS["failed_names"]:
        print("Failed:")
        for n in _RESULTS["failed_names"]:
            print(f"  - {n}")
    print("=" * 60)
    sys.exit(1 if _RESULTS["fail"] else 0)

   