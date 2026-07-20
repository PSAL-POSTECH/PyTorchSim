import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import argparse
from Simulator.simulator import TOGSimulator
device = torch.device("npu:0")
# ─────────────────────────────────────────────────────────────────────────────
# Optimized: Flash-Decode style — tile S upfront, batch in B dimension
# ─────────────────────────────────────────────────────────────────────────────

class GQADecodeOptimized(nn.Module):
    """Flash-Decode style GQA decode for multi-core NPU.

    Splits the KV-cache sequence into n_tiles chunks and folds them into the
    BMM batch dimension (B_total = H_kv × n_tiles).  Both the QK and SV
    matrix multiplications are issued as a *single* batched BMM with a short
    inner-K loop, so the NPU scheduler can distribute all B_total tiles across
    available cores simultaneously.

    Improvement over GQABaseline
    ─────────────────────────────
    Baseline QK : B=H_kv=1,      M=G, N=S(large), K=D  → 640 N-tile iters on 1 batch
    Optimized QK: B=H_kv*n_tiles, M=G, N=T(small), K=D  → n_tiles batch slots for cores

    Baseline SV : B=H_kv=1,      M=G, N=D, K=S   → K-loop=640, only 8 outer tiles
    Optimized SV: B=H_kv*n_tiles, M=G, N=D, K=T   → K-loop=T/TILE_K, n_tiles outer tiles

    Memory layout improvements
    ──────────────────────────
    • K/V tiles are generated with a single contiguous view+reshape (no mid-loop transpose).
    • Avoids materializing the full score tensor [H_kv, G, S] in DRAM before tiling.
    • Softmax intermediates are kept in smaller [B_total, G, T] buffers.

    Input conventions
    ─────────────────
        q : [H_kv, G, D]  – one decode-step query token per KV head
        k : [H_kv, S, D]  – KV-cache keys   (NOT pre-transposed)
        v : [H_kv, S, D]  – KV-cache values

    tile_size selection
    ───────────────────
        Ideal: tile_size = round_up(S * H_kv / num_cores, vpu_num_lanes)
        so that B_total ≈ num_cores.  Must also satisfy the SPAD budget:
            (G*T + T*D + G*D) * bytes ≤ spad_per_core   (for sub-tile occupancy)
        Default 512 works for (G=5, D=128, fp16, 16-lane × 8 KB/lane SPAD).
    """

    def __init__(self, tile_size: int = 512):
        super().__init__()
        self.tile_size = tile_size

    def forward(
        self,
        q: torch.Tensor,   # [H_kv, G, D]
        k: torch.Tensor,   # [H_kv, S, D]
        v: torch.Tensor,   # [H_kv, S, D]
        scale: float,
    ) -> torch.Tensor:
        H_kv, G, D = q.shape
        _, S, _    = k.shape
        T          = self.tile_size
        n_tiles    = (S + T - 1) // T
        pad_len    = n_tiles * T - S
        B_total    = H_kv * n_tiles

        # ── 1. Pad S → multiple of T ───────────────────────────────────────
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))   # [H_kv, S', D]
            v = F.pad(v, (0, 0, 0, pad_len))   # [H_kv, S', D]

        # ── 2. Tile K, V → [B_total, T, D]  (contiguous, no copy) ─────────
        # k is [H_kv, S', D]; view splits S' → n_tiles×T along dim-1
        k_tiles = k.view(H_kv, n_tiles, T, D).reshape(B_total, T, D)
        v_tiles = v.view(H_kv, n_tiles, T, D).reshape(B_total, T, D)

        # ── 3. Expand Q → [B_total, G, D] ─────────────────────────────────
        # expand: zero-copy view; reshape: contiguous copy (small: B_total*G*D elems)
        q_exp = q.unsqueeze(1).expand(H_kv, n_tiles, G, D).reshape(B_total, G, D)

        # ── 4. Batched QK BMM ──────────────────────────────────────────────
        # [B_total, G, D] × [B_total, D, T] → [B_total, G, T]
        # NPU mapping: B=B_total, M=G, N=T, K=D
        #   → outer tiles = B_total × M_tiles × N_tiles  (all parallelizable)
        #   → inner K-loop = D/TILE_K  (short, D=128)
        k_t    = k_tiles.transpose(1, 2)            # [B_total, D, T]
        scores = torch.bmm(q_exp, k_t) * scale      # [B_total, G, T]

        # ── 5. Tile-local softmax (fp32 accumulation) ──────────────────────
        # All ops are elementwise on [B_total, G, T] → torch.compile fuses them
        scores_f32 = scores.float()
        local_max  = scores_f32.amax(dim=-1, keepdim=True)  # [B_total, G, 1]
        local_exp  = (scores_f32 - local_max).exp()          # [B_total, G, T]
        local_sum  = local_exp.sum(dim=-1, keepdim=True)     # [B_total, G, 1]

        # ── 6. Batched SV BMM ──────────────────────────────────────────────
        # [B_total, G, T] × [B_total, T, D] → [B_total, G, D]
        # NPU mapping: B=B_total, M=G, N=D, K=T
        #   → outer tiles = B_total × M_tiles × N_tiles  (parallelizable)
        #   → inner K-loop = T/TILE_K  (controlled, T≪S)
        sv = torch.bmm(local_exp.to(q.dtype), v_tiles)     # [B_total, G, D]

        # ── 7. Online-softmax global reduction (elementwise, fused) ────────
        local_max = local_max.view(H_kv, n_tiles, G, 1)
        local_sum = local_sum.view(H_kv, n_tiles, G, 1)
        sv        = sv.view(H_kv, n_tiles, G, D)

        global_max    = local_max.amax(dim=1, keepdim=True)     # [H_kv, 1, G, 1]
        rescale       = (local_max - global_max).exp()           # [H_kv, n_tiles, G, 1]
        corrected_sv  = (sv        * rescale).sum(dim=1)         # [H_kv, G, D]
        corrected_sum = (local_sum * rescale).sum(dim=1)         # [H_kv, G, 1]

        return (corrected_sv / corrected_sum.clamp_min(1e-12)).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "LLAMA4_TP8": {
        "HEAD_DIM":     128,
        "NUM_HEADS":    5,    # = 40 total / TP8
        "NUM_KV_HEADS": 1,    # =  8 total / TP8
    },
    "QWEN3-235B_TP4": {
        "HEAD_DIM":     128,
        "NUM_HEADS": 16,
        "NUM_KV_HEADS": 1,
    },
    "GPT-OSS_TP1": {
        "HEAD_DIM":     64,
        "NUM_HEADS": 64,
        "NUM_KV_HEADS": 8,
    },
    "GPT-OSS_TP2": {
        "HEAD_DIM":     64,
        "NUM_HEADS": 32,
        "NUM_KV_HEADS": 4,
    },
    "GPT-OSS_TP4": {
        "HEAD_DIM":     64,
        "NUM_HEADS": 16,
        "NUM_KV_HEADS": 2,
    },
    "GPT-OSS_TP8": {
        "HEAD_DIM":     64,
        "NUM_HEADS":  8,
        "NUM_KV_HEADS": 1,
    },
}


def _make_inputs(cfg, seq_len, dtype):
    H_kv  = cfg["NUM_KV_HEADS"]
    G     = cfg["NUM_HEADS"] // cfg["NUM_KV_HEADS"]
    D     = cfg["HEAD_DIM"]
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(H_kv, G, D,        dtype=dtype)
    k = torch.randn(H_kv, seq_len, D,  dtype=dtype)   # NOT pre-transposed
    v = torch.randn(H_kv, seq_len, D,  dtype=dtype)
    return q, k, v, scale


def test_gqa_decode_optimized(model, device, seq_len: int = 10240, tile_size: int = 512):

    cfg = MODEL_CONFIGS[model] if model is not None else MODEL_CONFIGS["LLAMA4_TP8"]
    dtype = torch.float16

    model = GQADecodeOptimized(tile_size=tile_size).eval()

    # ── NPU run ────────────────────────────────────────────────────────────
    q, k, v, scale = _make_inputs(cfg, seq_len, dtype)
    model_dev      = model.to(device)
    compiled       = torch.compile(model_dev, dynamic=False)

    q_dev, k_dev, v_dev = q.to(device), k.to(device), v.to(device)
    with torch.no_grad():
        with TOGSimulator():
            out_dev = compiled(q_dev, k_dev, v_dev, scale=scale)

    # ── CPU reference ──────────────────────────────────────────────────────
    with torch.no_grad():
        out_cpu = model.cpu()(q, k, v, scale=scale)

    max_diff = (out_dev.cpu() - out_cpu).abs().max().item()

    with torch.no_grad():#CPU reference
        out_library = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True)

    max_diff_library = (out_library.cpu() - out_cpu).abs().max().item()

    print(f"[GQADecodeOptimized] seq_len={seq_len}, tile_size={tile_size}")
    print(f"  max |npu - cpu| = {max_diff:.6f}")
    print(f"  npu out max     = {out_dev.cpu().abs().max().item():.6f}")
    print(f"  cpu out max     = {out_cpu.abs().max().item():.6f}")
    print(f"  library out max = {out_library.abs().max().item():.6f}")
    print("  PASS" if max_diff < 0.05 else "  FAIL (diff too large)")




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Test GQA Attention Implementations")
    argparser.add_argument("--model", type=str, default="LLAMA4_TP8", choices=MODEL_CONFIGS.keys(), help="Model configuration to test")
    argparser.add_argument("--context_length", type=int, default=10240, help="Sequence length (context length) for the attention test")
    argparser.add_argument("--tile_size", type=int, default=4096, help="Tile size for the optimized attention implementation")
    args = argparser.parse_args()
    model = args.model
    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    sys.path.append(base_dir)
    test_gqa_decode_optimized(model=model, device=device, seq_len=args.context_length, tile_size=args.tile_size)
