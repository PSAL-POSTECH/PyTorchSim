import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import argparse


def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)


class GQAMultiheadAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    Query has num_heads, but key/value have num_kv_heads (num_kv_heads < num_heads).
    """
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, head_dim=None, bias=True, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        if head_dim is None:
            head_dim = embed_dim // num_heads
        assert embed_dim == num_heads * head_dim
        
        # If num_kv_heads is not specified, use num_heads (standard MHA)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        
        assert num_kv_heads <= num_heads
        assert embed_dim % num_kv_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        # QKV projection: Q has embed_dim, K and V have kv_embed_dim each
        kv_embed_dim = num_kv_heads * head_dim
        total_qkv_dim = embed_dim + 2 * kv_embed_dim
        
        self.qkv_proj = nn.Linear(embed_dim, total_qkv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        """
        Args:
            query: [batch, seq_len, embed_dim] or [seq_len, batch, embed_dim]
            key: optional, same shape as query
            value: optional, same shape as query
            attn_mask: optional attention mask
            need_weights: whether to return attention weights
        """
        # For compatibility with nn.MultiheadAttention API
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Handle batch_first vs batch_second
        if query.dim() == 3:
            batch_first = True
            batch_size, seq_len, _ = query.shape
        else:
            batch_first = False
            seq_len, batch_size, _ = query.shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project QKV
        # Use query for QKV projection (standard MHA/GQA pattern)
        qkv = self.qkv_proj(query)  # [batch, seq_len, total_qkv_dim]
        
        # Split into Q, K, V
        kv_embed_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :, :self.embed_dim]  # [batch, seq_len, embed_dim]
        k = qkv[:, :, self.embed_dim:self.embed_dim + kv_embed_dim]  # [batch, seq_len, kv_embed_dim]
        v = qkv[:, :, self.embed_dim + kv_embed_dim:]  # [batch, seq_len, kv_embed_dim]
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [batch, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)  # [batch, seq_len, num_kv_heads, head_dim]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)  # [batch, seq_len, num_kv_heads, head_dim]
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Scaled dot product attention with GQA support
        # enable_gqa=True allows different number of heads for Q vs K/V
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=(self.num_kv_heads < self.num_heads)
        )  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)  # [batch, seq_len, embed_dim]
        
        if not batch_first:
            output = output.transpose(0, 1)  # [seq_len, batch, embed_dim]
        
        if need_weights:
            # Compute attention weights for return
            # This is simplified - in practice you'd want the actual attention weights
            attn_weights = None
            return output, attn_weights
        else:
            return output


def test_gqa_attention(device, batch=1, seq_len=32, embed_dim=768, num_heads=12, num_kv_heads=4):
    """
    Test Grouped Query Attention (GQA) where num_kv_heads < num_heads.
    
    Args:
        device: target device
        batch: batch size
        seq_len: sequence length
        embed_dim: embedding dimension
        num_heads: number of query heads
        num_kv_heads: number of key/value heads (should be <= num_heads)
    """
    print(f"Testing GQA Attention (batch={batch}, seq_len={seq_len}, embed_dim={embed_dim}, "
          f"num_heads={num_heads}, num_kv_heads={num_kv_heads})")
    
    # Create GQA model
    gqa = GQAMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        bias=True,
        dropout=0.0
    ).eval()
    
    # Initialize weights
    torch.nn.init.normal_(gqa.qkv_proj.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(gqa.qkv_proj.bias, mean=0.0, std=0.02)
    torch.nn.init.normal_(gqa.out_proj.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(gqa.out_proj.bias, mean=0.0, std=0.02)
    
    # Create input
    x = torch.randn(batch, seq_len, embed_dim)
    query = x.clone()
    key = x.clone()
    value = x.clone()
    
    # Run on custom device
    gqa_device = gqa.to(device)
    q1, k1, v1 = query.to(device), key.to(device), value.to(device)
    
    compiled_gqa = torch.compile(gqa_device, dynamic=False)
    with torch.no_grad():
        out_device = compiled_gqa(q1, k1, v1)
    
    # Run on CPU
    gqa_cpu = gqa.cpu()
    q2, k2, v2 = query.cpu(), key.cpu(), value.cpu()
    with torch.no_grad():
        out_cpu = gqa_cpu(q2, k2, v2)
    
    test_result("GQA Attention", out_device, out_cpu)
    print("Max diff > ", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("GQA Attention Simulation Done")


def test_standard_mha_via_gqa(device, batch=1, seq_len=32, embed_dim=768, num_heads=12):
    """
    Test standard Multi-Head Attention using GQA with num_kv_heads == num_heads.
    This should behave the same as standard MHA.
    """
    print(f"Testing Standard MHA via GQA (batch={batch}, seq_len={seq_len}, "
          f"embed_dim={embed_dim}, num_heads={num_heads})")
    
    test_gqa_attention(device, batch, seq_len, embed_dim, num_heads, num_kv_heads=num_heads)


def test_repeat_interleave_compilation(device, batch=1, seq_len=32, embed_dim=768, num_heads=12, num_kv_heads=4):
    """
    Test that repeat_interleave operation compiles and works correctly using scaled_dot_product_attention implementation.
    
    This test uses the exact implementation from F.scaled_dot_product_attention to verify
    that repeat_interleave works correctly when enable_gqa=True.
    
    Args:
        device: target device
        batch: batch size
        seq_len: sequence length
        embed_dim: embedding dimension
        num_heads: number of query heads
        num_kv_heads: number of key/value heads (should be < num_heads)
    """
    import math
    
    print(f"Testing repeat_interleave compilation using scaled_dot_product_attention implementation "
          f"(batch={batch}, seq_len={seq_len}, embed_dim={embed_dim}, "
          f"num_heads={num_heads}, num_kv_heads={num_kv_heads})")
    
    head_dim = embed_dim // num_heads
    assert num_kv_heads < num_heads, "num_kv_heads must be less than num_heads for GQA"
    
    # Create Q, K, V tensors
    # Q: [batch, num_heads, seq_len, head_dim]
    # K, V: [batch, num_kv_heads, seq_len, head_dim]
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    
    # Move to device
    q_device = q.to(device)
    k_device = k.to(device)
    v_device = v.to(device)
    
    # Implementation from F.scaled_dot_product_attention
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight, value, attn_weight @ value
    
    # Compile the function
    compiled_attn = torch.compile(scaled_dot_product_attention, dynamic=False)
    
    # Run on custom device with enable_gqa=True
    with torch.no_grad():
        output_device = compiled_attn(q_device, k_device, v_device, 
                                      attn_mask=None, dropout_p=0.0, 
                                      is_causal=False, scale=None, enable_gqa=True)
    
    # Run on CPU for comparison
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    with torch.no_grad():
        output_cpu = scaled_dot_product_attention(q_cpu, k_cpu, v_cpu,
                                                  attn_mask=None, dropout_p=0.0,
                                                  is_causal=False, scale=None, enable_gqa=True)
    
    # Compare results
    test_result("repeat_interleave in scaled_dot_product_attention", output_device[0], output_cpu[0])
    print("Max diff > ", torch.max(torch.abs(output_device[0].cpu() - output_cpu[0])))
    test_result("repeat_interleave in scaled_dot_product_attention", output_device[1], output_cpu[1])
    print("Max diff > ", torch.max(torch.abs(output_device[1].cpu() - output_cpu[1])))
    test_result("repeat_interleave in scaled_dot_product_attention", output_device[2], output_cpu[2])
    print("Max diff > ", torch.max(torch.abs(output_device[2].cpu() - output_cpu[2])))
    print("repeat_interleave compilation test Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="npu", help="Device to use")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--num_kv_heads", type=int, default=4, help="Number of key/value heads")
    parser.add_argument("--test_standard", action="store_true", help="Also test standard MHA via GQA")
    parser.add_argument("--test_repeat_interleave", action="store_true", help="Test repeat_interleave compilation")
    
    args = parser.parse_args()

    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
    device = torch.device("npu:0")
    
    test_repeat_interleave_compilation(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads
    )
    
    # Test GQA
    test_gqa_attention(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads
    )
    
    # Optionally test standard MHA via GQA
    # if args.test_standard:
    #    test_standard_mha_via_gqa(
    #        device=args.device,
    #        batch=args.batch,
    #        seq_len=args.seq_len,
    #        embed_dim=args.embed_dim,
    #        num_heads=args.num_heads
    #    )
