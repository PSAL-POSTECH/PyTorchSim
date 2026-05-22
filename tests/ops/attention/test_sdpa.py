import sys
import os
import torch
import torch._dynamo
import torch.nn.functional as F

base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.append(base_dir)

device = torch.device("npu:0")

# ---------------------------------------------------------------------------
# Default sweep configs - edit here to change what gets tested
# ---------------------------------------------------------------------------
SDPA_DEFAULTS = dict(
    n_batch_list  = [1, 4, 8, 16],
    n_head_list   = [4, 6, 8, 12],
    n_token_list  = [128, 256, 512, 1024],
    head_dim_list = [32, 64, 128],
    is_causal     = False,
)

GQA_DEFAULTS = dict(
    batch_list      = [1],
    num_kv_heads    = 1,
    gqa_ratios      = [4, 5, 8, 16],   # Hq = ratio * num_kv_heads
    seq_len_list    = [128, 256, 1024],
    head_dim_list   = [64, 128],
    query_len       = 1,               # decode shape: Lq == 1
    is_causal       = True,
)


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


def assert_close(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    msg = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
    else:
        print(f"[FAIL] {name}")
        print("  device out:", out.cpu())
        print("  cpu    out:", cpu_out)
        exit(1)


def _run_sdpa(device, q, k, v, **kwargs):
    """Compile and run SDPA on device; return result on device."""
    opt_fn = torch.compile(dynamic=False)(F.scaled_dot_product_attention)
    return opt_fn(q.to(device), k.to(device), v.to(device), **kwargs)


def _cpu_sdpa(q, k, v, **kwargs):
    """Run reference SDPA on CPU."""
    return F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sdpa(
    device,
    n_batch_list  = SDPA_DEFAULTS["n_batch_list"],
    n_head_list   = SDPA_DEFAULTS["n_head_list"],
    n_token_list  = SDPA_DEFAULTS["n_token_list"],
    head_dim_list = SDPA_DEFAULTS["head_dim_list"],
    is_causal     = SDPA_DEFAULTS["is_causal"],
):
    torch.manual_seed(0)
    sdpa_kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=is_causal)

    for B in n_batch_list:
        for H in n_head_list:
            for S in n_token_list:
                for D in head_dim_list:
                    clear_caches()
                    q = torch.rand(B, H, S, D, dtype=torch.float32)
                    k = torch.rand(B, H, S, D, dtype=torch.float32)
                    v = torch.rand(B, H, S, D, dtype=torch.float32)

                    out     = _run_sdpa(device, q, k, v, **sdpa_kwargs)
                    cpu_out = _cpu_sdpa(q, k, v, **sdpa_kwargs)

                    assert_close(f"SDPA(B:{B}, H:{H}, S:{S}, D:{D})", out, cpu_out)

    print("All SDPA tests passed!")


def test_gqa(
    device,
    batch_list   = GQA_DEFAULTS["batch_list"],
    num_kv_heads = GQA_DEFAULTS["num_kv_heads"],
    gqa_ratios   = GQA_DEFAULTS["gqa_ratios"],
    seq_len_list = GQA_DEFAULTS["seq_len_list"],
    head_dim_list= GQA_DEFAULTS["head_dim_list"],
    query_len    = GQA_DEFAULTS["query_len"],
    is_causal    = GQA_DEFAULTS["is_causal"],
):
    """
    GQA sweep: q shape (B, Hq, Lq, D), kv shape (B, H, S, D).
    Hq = ratio * num_kv_heads for each ratio in gqa_ratios.
    """
    torch.manual_seed(0)
    sdpa_kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=is_causal, enable_gqa=True)

    for B in batch_list:
        for S in seq_len_list:
            for D in head_dim_list:
                for ratio in gqa_ratios:
                    Hq = ratio * num_kv_heads
                    clear_caches()
                    q = torch.rand(B, Hq, query_len, D, dtype=torch.float32)
                    k = torch.rand(B, num_kv_heads, S, D, dtype=torch.float32)
                    v = torch.rand(B, num_kv_heads, S, D, dtype=torch.float32)

                    out     = _run_sdpa(device, q, k, v, **sdpa_kwargs)
                    cpu_out = _cpu_sdpa(q, k, v, **sdpa_kwargs)

                    assert_close(
                        f"GQA(B:{B}, Hq:{Hq}, H:{num_kv_heads}, S:{S}, D:{D})",
                        out, cpu_out,
                    )

    print("All GQA tests passed!")


if __name__ == "__main__":
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        test_sdpa(device)
    #test_gqa(device)

    # Example: quick single-config run
    # test_gqa(device, batch_list=[1], gqa_ratios=[5], seq_len_list=[32], head_dim_list=[128])
