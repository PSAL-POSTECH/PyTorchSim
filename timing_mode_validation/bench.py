"""One parameterised microbenchmark per operation, for cycle validation.

``--op`` names the operation and ``--size`` its dimensions; every row of
``experiments/artifact/timing_cases.csv`` is one invocation of this file.
"""

import argparse
import math
import os
import sys

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.insert(0, base_path)

import torch


def _default_config():
    """v6e's timing config when this tree has it, else the v3 one."""
    v6e = f'{base_path}/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_timing_only.yml'
    v3 = f'{base_path}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml'
    return v6e if os.path.exists(v6e) else v3


config = os.environ.get('TOGSIM_CONFIG', _default_config())
os.environ['TOGSIM_CONFIG'] = config

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def widen_result(fn):
    """Keep the operands narrow and hand the result back as fp32.

    The only narrow-float form this backend carries is the one
    kernels/coverage/mixed/mixed_gemm_fp16.py pins: fp16 operands go straight
    into `tl.dot`, the accumulator and the store are fp32, and nothing touches
    fp16 in the vector unit -- a convert there scalarises and stage 6 refuses it.
    """
    def call(*args):
        out = fn(*args)
        if torch.is_tensor(out) and out.is_floating_point():
            return out.float()
        return out
    return call


def _rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


# --- matmul family ----------------------------------------------------------

def gemm(M, K, N, dt):
    """A @ B, the projection shape every transformer layer is made of."""
    return (lambda a, b: torch.matmul(a, b)), [_rand(M, K, dtype=dt), _rand(K, N, dtype=dt)]


def gemm_bias(M, K, N, dt):
    """A @ B + bias, the shape a projection with bias takes."""
    return ((lambda a, b, c: torch.addmm(c, a, b)),
            [_rand(M, K, dtype=dt), _rand(K, N, dtype=dt), _rand(N, dtype=dt)])


def bmm(B, M, K, N, dt):
    """Batched A @ B, one batch per attention head."""
    return (lambda a, b: torch.bmm(a, b)), [_rand(B, M, K, dtype=dt), _rand(B, K, N, dtype=dt)]


def attn_qk(B, S, D, dt):
    """The score matmul alone: [B,S,D] @ [B,D,S]."""
    return ((lambda q, k: torch.matmul(q, k.transpose(-2, -1))),
            [_rand(B, S, D, dtype=dt), _rand(B, S, D, dtype=dt)])


def attn_pv(B, S, D, dt):
    """The context matmul alone: [B,S,S] @ [B,S,D]."""
    return (lambda p, v: torch.matmul(p, v)), [_rand(B, S, S, dtype=dt), _rand(B, S, D, dtype=dt)]


def attention(B, S, D, dt):
    """Score, softmax and context together -- the same math as attention.py."""
    def fn(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(scores.softmax(dim=-1), v)
    return fn, [_rand(B, S, D, dtype=dt) for _ in range(3)]


def attn_causal(B, S, D, dt):
    """Attention with the additive causal mask the models actually build."""
    def fn(q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)) + mask
        return torch.matmul(scores.softmax(dim=-1), v)
    mask = torch.triu(torch.full((S, S), float("-inf"), dtype=dt), diagonal=1)
    return fn, [_rand(B, S, D, dtype=dt) for _ in range(3)] + [mask]


def attn_decode(B, S_kv, D, dt):
    """One query against a KV cache: the matrix-vector regime of serving."""
    def fn(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(scores.softmax(dim=-1), v)
    return fn, [_rand(B, 1, D, dtype=dt), _rand(B, S_kv, D, dtype=dt), _rand(B, S_kv, D, dtype=dt)]


def gqa_attn(Hq, Hkv, S, D, dt):
    """Attention with repeat_kv in the graph, which is what GQA costs."""
    n_rep = Hq // Hkv

    def fn(q, k, v):
        k2 = k.repeat_interleave(n_rep, dim=0)
        v2 = v.repeat_interleave(n_rep, dim=0)
        scores = torch.matmul(q, k2.transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(scores.softmax(dim=-1), v2)
    return fn, [_rand(Hq, S, D, dtype=dt), _rand(Hkv, S, D, dtype=dt), _rand(Hkv, S, D, dtype=dt)]


# --- normalisation, activation, elementwise ---------------------------------

def rmsnorm(R, C, dt):
    """RMSNorm over the last axis -- 45 of the 68 captured models use it."""
    def fn(x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + 1e-6) * w
    return fn, [_rand(R, C, dtype=dt), _rand(C, dtype=dt)]


def layernorm(R, C, dt):
    """LayerNorm over the last axis, the encoder-side norm."""
    return ((lambda x, w, b: torch.nn.functional.layer_norm(x, (x.size(-1),), w, b)),
            [_rand(R, C, dtype=dt), _rand(C, dtype=dt), _rand(C, dtype=dt)])


def softmax(R, C, dt):
    """Row softmax, the shape attention scores arrive in."""
    return (lambda x: torch.softmax(x, dim=-1)), [_rand(R, C, dtype=dt)]


def softmax3d(B, S, C, dt):
    """Softmax on a [head, query, key] score tensor, without the matmuls."""
    return (lambda x: torch.softmax(x, dim=-1)), [_rand(B, S, C, dtype=dt)]


def swiglu(R, C, dt):
    """silu(gate) * up, the elementwise half of a gated MLP."""
    return ((lambda a, b: torch.nn.functional.silu(a) * b),
            [_rand(R, C, dtype=dt), _rand(R, C, dtype=dt)])


def gelu(R, C, dt):
    """GELU on an MLP-width tensor."""
    return (lambda x: torch.nn.functional.gelu(x, approximate="tanh")), [_rand(R, C, dtype=dt)]


def residual(R, C, dt):
    """A plain residual add, the cheapest DRAM-bound shape in a layer."""
    return (lambda a, b: a + b), [_rand(R, C, dtype=dt), _rand(R, C, dtype=dt)]


def rope(B, H, S, D, dt):
    """Rotary embedding applied to one projection, rotate_half form."""
    def fn(x, cos, sin):
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin
    return fn, [_rand(B, H, S, D, dtype=dt), _rand(1, 1, S, D, dtype=dt), _rand(1, 1, S, D, dtype=dt)]


def rope_outer(S, D, dt):
    """The inv_freq outer product: a matmul whose K is 1."""
    return ((lambda p, f: torch.matmul(p, f)),
            [_rand(S, 1, dtype=dt), _rand(1, D, dtype=dt)])


def reduce_sum(R, C, dt):
    """Row-wise sum, the reduction shape a norm and a softmax share."""
    return (lambda x: x.sum(-1)), [_rand(R, C, dtype=dt)]


# --- gather, scatter, routing ------------------------------------------------

def embedding(V, H, T, dt):
    """Token embedding: T rows gathered out of a [V, H] table."""
    ids = torch.randint(0, V, (1, T), dtype=torch.int64)
    return (lambda w, i: torch.nn.functional.embedding(i, w)), [_rand(V, H, dtype=dt), ids]


def topk(T, E, K, dt):
    """Top-k over per-token expert logits, the MoE router's first step."""
    return (lambda x: torch.topk(x, K, dim=-1)), [_rand(T, E, dtype=dt)]


def moe_route(T, E, K, dt):
    """Router softmax, top-k and renormalise -- the whole gate."""
    def fn(x):
        w = torch.softmax(x, dim=-1)
        v, i = torch.topk(w, K, dim=-1)
        return v / v.sum(-1, keepdim=True), i
    return fn, [_rand(T, E, dtype=dt)]


def dispatch(T, K, H, dt):
    """Gather tokens to experts and scatter the results back."""
    idx = torch.randint(0, T, (T * K,), dtype=torch.int64)

    def fn(x, i):
        picked = torch.index_select(x, 0, i)
        out = torch.zeros_like(x)
        return out.index_add(0, i, picked)
    return fn, [_rand(T, H, dtype=dt), idx]


def sort1d(T, E, K, dt):
    """The router's argsort of flattened top-k ids -- what counting sort replaces.

    Sorting a bare tensor does not compile: the rewrite only fires when the
    keys trace back to a topk output, which is what proves their range.
    """
    def fn(x):
        _, idx = torch.topk(x, K, dim=-1)
        return torch.argsort(idx.view(-1))
    return fn, [_rand(T, E, dtype=dt)]


# --- convolution -------------------------------------------------------------

def conv(N, H, W, C, K, R, stride, pad, dt):
    """A dense 2-D convolution, the ResNet-shaped baseline."""
    return ((lambda x, w: torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)),
            [_rand(N, C, H, W, dtype=dt), _rand(K, C, R, R, dtype=dt)])


def dwconv(N, C, H, W, R, stride, dt):
    """Depthwise convolution: groups == channels, one filter per channel."""
    return ((lambda x, w: torch.nn.functional.conv2d(x, w, stride=stride, padding=R // 2, groups=C)),
            [_rand(N, C, H, W, dtype=dt), _rand(C, 1, R, R, dtype=dt)])


def pwconv(N, C, H, W, K, dt):
    """Pointwise 1x1 convolution, the most common conv node in the census."""
    return ((lambda x, w: torch.nn.functional.conv2d(x, w)),
            [_rand(N, C, H, W, dtype=dt), _rand(K, C, 1, 1, dtype=dt)])


def patch_embed(N, C, H, K, P, dt):
    """The vision patch embedding: kernel == stride, no overlap."""
    return ((lambda x, w: torch.nn.functional.conv2d(x, w, stride=P)),
            [_rand(N, C, H, H, dtype=dt), _rand(K, C, P, P, dtype=dt)])


def conv1d_causal(N, C, L, R, dt):
    """The depthwise causal Conv1d a linear-attention block puts in the loop."""
    return ((lambda x, w: torch.nn.functional.conv1d(x, w, padding=R - 1, groups=C)),
            [_rand(N, C, L, dtype=dt), _rand(C, 1, R, dtype=dt)])


def convtranspose(N, C, H, W, K, R, stride, dt):
    """A transposed convolution, which the frontend rewrites as a direct one."""
    return ((lambda x, w: torch.nn.functional.conv_transpose2d(x, w, stride=stride)),
            [_rand(N, C, H, W, dtype=dt), _rand(C, K, R, R, dtype=dt)])


def maxpool(N, C, H, W, k, stride, dt):
    """Max pooling, the YOLO SPPF and ResNet stem shape."""
    return ((lambda x: torch.nn.functional.max_pool2d(x, k, stride=stride, padding=k // 2)),
            [_rand(N, C, H, W, dtype=dt)])


# --- blocks ------------------------------------------------------------------

def mlp_swiglu(S, hidden, interm, dt):
    """A whole gated MLP: three GEMMs and the elementwise between them."""
    def fn(x, wg, wu, wd):
        return torch.matmul(torch.nn.functional.silu(torch.matmul(x, wg)) * torch.matmul(x, wu), wd)
    return fn, [_rand(S, hidden, dtype=dt), _rand(hidden, interm, dtype=dt),
                _rand(hidden, interm, dtype=dt), _rand(interm, hidden, dtype=dt)]

def attn_block(S, hidden, heads, kv_heads, dt):
    """One attention block end to end: projections, GQA repeat, scores, context."""
    D = hidden // heads
    n_rep = heads // kv_heads

    def fn(x, wq, wk, wv, wo):
        q = torch.matmul(x, wq).view(S, heads, D).transpose(0, 1)
        k = torch.matmul(x, wk).view(S, kv_heads, D).transpose(0, 1)
        v = torch.matmul(x, wv).view(S, kv_heads, D).transpose(0, 1)
        k = k.repeat_interleave(n_rep, dim=0)
        v = v.repeat_interleave(n_rep, dim=0)
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        o = torch.matmul(s.softmax(dim=-1), v)
        return torch.matmul(o.transpose(0, 1).reshape(S, hidden), wo)
    return fn, [_rand(S, hidden, dtype=dt), _rand(hidden, hidden, dtype=dt),
                _rand(hidden, kv_heads * D, dtype=dt), _rand(hidden, kv_heads * D, dtype=dt),
                _rand(hidden, hidden, dtype=dt)]


def ffn_gated(S, hidden, interm, dt):
    """The decoder FFN as silicon ran it: gate and up are ONE [h, 2*interm] weight.

    Not mlp_swiglu, which keeps them as two GEMMs -- the measured program
    (tpuv6e_validation code/harness/tpu_lut_sweep_v2.py, op `ffn`) multiplies once
    and slices, so its second GEMM waits on a slice rather than on its own matmul.
    """
    def fn(x, wgu, wd):
        gu = torch.matmul(x, wgu)
        g, u = gu[:, :interm], gu[:, interm:]
        return torch.matmul(torch.nn.functional.silu(g) * u, wd)
    return fn, [_rand(S, hidden, dtype=dt), _rand(hidden, 2 * interm, dtype=dt),
                _rand(interm, hidden, dtype=dt)]


def ffn_erf(S, hidden, interm, dt):
    """The encoder FFN that represents serving: two GEMMs with bias, erf GELU between.

    erf, not tanh: on silicon the two lowerings differ by 2.1-3.3x because
    `gelu(approximate=False)` expands chlo.erfc into ~66 ops per element while the
    erf form lowers natively (gate g4 in the measurement tree).
    """
    def fn(x, w1, b1, w2, b2):
        y = torch.matmul(x, w1) + b1
        return torch.matmul(0.5 * y * (1 + torch.erf(y / math.sqrt(2))), w2) + b2
    return fn, [_rand(S, hidden, dtype=dt), _rand(hidden, interm, dtype=dt),
                _rand(interm, dtype=dt), _rand(interm, hidden, dtype=dt),
                _rand(hidden, dtype=dt)]


def attn_enc_full(Hq, S, D, dt):
    """Encoder self-attention as silicon ran it: non-causal, materialised scores.

    This is XLA's own path, not a flash kernel: splash needs d%128==0 and q>=128, and
    the measured encoders are d=64/32, so the whole [H, S, S] score tensor is built.
    """
    def fn(q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        return torch.matmul(s.softmax(dim=-1), v)
    return fn, [_rand(Hq, S, D, dtype=dt), _rand(Hq, S, D, dtype=dt),
                _rand(Hq, S, D, dtype=dt)]


def attn_dec_gqa(Hq, Hkv, q_len, kv_len, D, dt):
    """The OPERATION ragged_paged_attention performs, written plainly.

    GQA over a KV cache with the q block at the LAST q_len positions of kv. Silicon
    runs it as a paged kernel with online softmax; this materialises the scores, so a
    comparison prices two algorithms for one operation and has to say so.
    """
    n_rep = Hq // Hkv

    def fn(q, k, v, *rest):
        k2 = k.repeat_interleave(n_rep, dim=0)
        v2 = v.repeat_interleave(n_rep, dim=0)
        s = torch.matmul(q, k2.transpose(-2, -1)) / math.sqrt(D)
        if rest:
            s = s + rest[0]
        return torch.matmul(s.softmax(dim=-1), v2)

    t = [_rand(Hq, q_len, D, dtype=dt), _rand(Hkv, kv_len, D, dtype=dt),
         _rand(Hkv, kv_len, D, dtype=dt)]
    if q_len > 1:            # at q_len 1 the query sees all of kv, so there is no mask
        qi = torch.arange(q_len).unsqueeze(1) + (kv_len - q_len)
        kj = torch.arange(kv_len).unsqueeze(0)
        t.append(torch.where(qi >= kj, 0.0, float("-inf")).to(dt))
    return fn, t


OPS = {
    "gemm": gemm, "gemm_bias": gemm_bias, "bmm": bmm,
    "attn_qk": attn_qk, "attn_pv": attn_pv, "attention": attention,
    "attn_causal": attn_causal, "attn_decode": attn_decode, "gqa_attn": gqa_attn,
    "rmsnorm": rmsnorm, "layernorm": layernorm, "softmax": softmax, "softmax3d": softmax3d,
    "swiglu": swiglu, "gelu": gelu, "residual": residual, "reduce_sum": reduce_sum,
    "rope": rope, "rope_outer": rope_outer,
    "embedding": embedding, "topk": topk, "moe_route": moe_route, "dispatch": dispatch,
    "sort1d": sort1d,
    "conv": conv, "dwconv": dwconv, "pwconv": pwconv, "patch_embed": patch_embed,
    "conv1d_causal": conv1d_causal, "convtranspose": convtranspose, "maxpool": maxpool,
    "mlp_swiglu": mlp_swiglu, "attn_block": attn_block,
    "ffn_gated": ffn_gated, "ffn_erf": ffn_erf,
    "attn_enc_full": attn_enc_full, "attn_dec_gqa": attn_dec_gqa,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One op, one shape, one cycle count")
    ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--size", nargs="+", type=int, required=True)
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    args = ap.parse_args()

    dt = DTYPES[args.dtype]
    fn, tensors = OPS[args.op](*args.size, dt)
    if dt is not torch.float32:
        fn = widen_result(fn)

    device = torch.device("npu:0")
    torch.manual_seed(0)
    tensors = [t.to(device=device) for t in tensors]
    opt_fn = torch.compile(dynamic=False)(fn)

    with torch.npu.simulator(config_path=config), torch.no_grad():
        torch.npu.launch_model(opt_fn, *tensors, stream_index=0, timestamp=0)
        torch.npu.synchronize()
    print(f"{args.op} {'x'.join(map(str, args.size))} {args.dtype} Simulation Done")
