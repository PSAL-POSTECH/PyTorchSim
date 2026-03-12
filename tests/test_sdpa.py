import sys
import math
import torch
import inspect
from typing import List
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel 
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch._inductor.decomposition import decompositions

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        pass
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_scaled_dot_product_attention(device, backends="flash"):
    torch.manual_seed(0)
    n_batch_list = [1, 4, 8, 16]
    n_head_list = [1, 4, 8, 12]
    n_token_list = [128, 256, 512, 1024]
    head_dim_list = [32, 64, 128]

    for n_batch in n_batch_list:
        for n_head in n_head_list:
            for n_token in n_token_list:
                for head_dim in head_dim_list:
                    # Inputs
                    clear_caches()
                    query = torch.rand(n_batch, n_head, n_token, head_dim, dtype=torch.float32)
                    key = torch.rand(n_batch, n_head, n_token, head_dim, dtype=torch.float32)
                    value = torch.rand(n_batch, n_head, n_token, head_dim, dtype=torch.float32)

                    # With NPU
                    query = query.to(device=device)
                    key = key.to(device=device)
                    value = value.to(device=device)

                    opt_fn = torch.compile(dynamic=False)(F.scaled_dot_product_attention)
                    out = opt_fn(query, key, value)
                    out = out.to(device)

                    # With CPU
                    cpu_device = torch.device('cpu')
                    query = query.to(device=cpu_device)
                    key = key.to(device=cpu_device)
                    value = value.to(device=cpu_device)
                    cpu_out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

                    name = f"SDPA(n_batch: {n_batch}, n_head: {n_head}, n_token: {n_token}, head_dim: {head_dim})"
                    test_result(name, out, cpu_out)
    
    print("All tests passed!")

def clear_caches():
    import os
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch._inductor.codecache import FxGraphCache
    AOTAutogradCache.clear()
    torch._dynamo.reset()
    os.environ["TORCHINDUCTOR_CACHE"] = "0"
    FxGraphCache.clear()

if __name__ == "__main__":    
    device = torch.device('npu:0')
    test_scaled_dot_product_attention(device, backends="flash")
    