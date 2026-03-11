
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import os
import sys
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend
import math
base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.append(base_dir)

os.environ["TOGSIM_CONFIG"] = f"{base_dir}/hhk_experiments/ndp_config.yml"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
config = f"{base_dir}/hhk_experiments/ndp_config.yml"
from Scheduler.scheduler import PyTorchSimRunner
device = PyTorchSimRunner.setup_device().custom_device()


def tiled_softmax(score: torch.Tensor, tile_size: int = 512, acc_dtype=torch.float32) -> torch.Tensor:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if score.ndim != 3:
        raise ValueError("score must be 3D tensor")
    bsz, group_size, seq_len = score.shape

    n_tiles = (seq_len + tile_size - 1) // tile_size
    pad_len = n_tiles * tile_size - seq_len


    if pad_len == 0:
        score_tiles = score.view(bsz, group_size, n_tiles, tile_size)
    else:
        score_padded = F.pad(score, (0, pad_len), value=-torch.inf)
        score_tiles = score_padded.view(bsz, group_size, n_tiles, tile_size)

    local_max = score_tiles.amax(dim=-1, keepdim=True)
    global_max = local_max.amax(dim=-2, keepdim=True)
    exp_tiles = torch.exp((score_tiles - global_max).to(acc_dtype))
    local_sum = exp_tiles.sum(dim=-1, keepdim=True)
    global_sum = local_sum.sum(dim=-2, keepdim=True).clamp_min(1e-12)

    prob_tiles = exp_tiles / global_sum
    probs = prob_tiles.reshape(bsz, group_size, n_tiles * tile_size)[..., :seq_len]
    return probs.to(score.dtype)

class GQAImplementation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, scale=0.006, enable_gqa=True, dtype=torch.float16, tile_size=512):
        # with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        score = torch.bmm(q, k) * scale
        attn_prob = tiled_softmax(score, tile_size=tile_size, acc_dtype=torch.float32)
        bmm_dtype = score.dtype
        attn_output = torch.bmm(attn_prob.to(bmm_dtype), v.to(bmm_dtype))
        attn_output = attn_output.to(dtype)
        return attn_output


def test_gqa_attention(device, batch=1, seq_len=10240):
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
    
    # Create GQA model
    gqa = GQAImplementation().eval()

    gqa = gqa.to(dtype=torch.float32)
    gqa_device = gqa.to(device)
    configs = {
        'LLAMA4_MODEL': { #TP8
            'HEAD_DIM': 128,
            'NUM_HEADS': 5,
            'NUM_KV_HEADS': 1,
        }
    }
    HEAD_DIM = configs['LLAMA4_MODEL']['HEAD_DIM']
    NUM_HEADS = configs['LLAMA4_MODEL']['NUM_HEADS']
    NUM_KV_HEADS = configs['LLAMA4_MODEL']['NUM_KV_HEADS']
    GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
    dtype = torch.float16
    query = torch.randn(NUM_KV_HEADS, GROUP_SIZE, HEAD_DIM, dtype=dtype)
    key = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=dtype)
    key = key.transpose(-2, -1)
    scale = 1 / math.sqrt(HEAD_DIM)
    value = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=dtype)
    # Run on custom devic
    
    q1, k1, v1 = query.to(device), key.to(device), value.to(device)
    compiled_gqa = torch.compile(gqa_device, dynamic=False)
    with torch.no_grad():
        with TOGSimulator(config_path=config):
            out_device = compiled_gqa(q1, k1, v1, scale=scale)
    
    # Run on CPU
    gqa_cpu = gqa.cpu()
    q2, k2, v2 = query.cpu(), key.cpu(), value.cpu()
    with torch.no_grad():
        out_cpu = gqa_cpu(q2, k2, v2, scale=scale)

    print("Max diff > ", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("Output device max > ", torch.max(torch.abs(out_device.cpu())))
    print("Output cpu max > ", torch.max(torch.abs(out_cpu)))
    print("GQA Attention Simulation Done")

if __name__ == "__main__":
    test_gqa_attention(device=device, batch=1, seq_len=102400)