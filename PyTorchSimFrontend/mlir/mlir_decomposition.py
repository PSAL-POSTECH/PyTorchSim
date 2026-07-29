import math
import operator
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch._inductor.decomposition import register_decomposition

aten = torch.ops.aten  # only for @register_decomposition target


def _pair_2d(seq: Sequence[int]) -> Tuple[int, int]:
    if len(seq) == 1:
        v = int(seq[0])
        return v, v
    return int(seq[0]), int(seq[1])


def _int_eq(x, v: int) -> bool:
    try:
        return int(x) == v
    except (TypeError, ValueError):
        return False


def _can_rewrite_pointwise_conv_on_1x1_spatial_to_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> bool:
    """
    Whether this ``aten.convolution`` is **exactly** ``F.linear`` on ``[N, C]`` (then reshaped
    to ``[N, C_out, 1, 1]``): 1x1 kernel, spatial size 1x1, ``groups==1``, stride 1, no padding,
    dilation 1 (typical SE line after global pool).

    If True, use ``_apply_pointwise_conv_on_1x1_spatial_as_linear``; if False, keep normal conv.
    """
    if transposed or input.dim() != 4 or weight.dim() != 4:
        return False
    if groups != 1:
        return False
    if not (
        _int_eq(input.shape[2], 1)
        and _int_eq(input.shape[3], 1)
        and _int_eq(weight.shape[2], 1)
        and _int_eq(weight.shape[3], 1)
    ):
        return False

    sh, sw = _pair_2d(stride)
    ph, pw = _pair_2d(padding)
    dh, dw = _pair_2d(dilation)
    if sh != 1 or sw != 1 or ph != 0 or pw != 0 or dh != 1 or dw != 1:
        return False
    if len(output_padding) and any(not _int_eq(o, 0) for o in output_padding):
        return False

    _, cin, _, _ = input.shape
    _, cin_w, _, _ = weight.shape
    try:
        if int(cin_w) != int(cin):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _apply_pointwise_conv_on_1x1_spatial_as_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Same numerics as ``convolution``; call only when ``_can_rewrite_...`` is True."""
    n, cin, _, _ = input.shape
    cout, _, _, _ = weight.shape
    x = input.reshape(n, cin)
    w = weight.reshape(cout, cin)
    return F.linear(x, w, bias).reshape(n, cout, 1, 1)


def _group_conv_cin1_cout1(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
) -> torch.Tensor:
    """
    Grouped conv with ``Cin//groups == 1`` and ``Cout//groups == 1`` (input ``[N,G,H,W]``, weight ``[G,1,Kh,Kw]``).

    1. Symmetric spatial padding on the input.
    2. For each kernel position ``(kh, kw)``, gather the output grid from the padded tensor and
       multiply by ``weight[:, 0, kh, kw]`` (broadcast over ``N``), then sum over ``(kh, kw)``.

    Note
    ----
    This is not a performance-optimized kernel: it is explicit gather–multiply–accumulate over
    kernel elements. For competitive performance, add a dedicated template (or fused) kernel
    instead of relying on this decomposition.
    """
    n, c_in, _, _ = input.shape
    # PyTorch layout: ``[Cout, Cin/groups, Kh, Kw]`` i.e. ``[G, 1, Kh, Kw]`` here.
    c_out, cin_pg, kh, kw = weight.shape
    g = groups
    assert c_in == g and c_out == g and cin_pg == 1, (c_in, c_out, cin_pg, g)

    sh, sw = _pair_2d(stride)
    ph, pw = _pair_2d(padding)
    d_h, d_w = _pair_2d(dilation)

    # (left, right, top, bottom) for last two dims
    x_pad = F.pad(input, (pw, pw, ph, ph))
    _, _, hp, wp = x_pad.shape

    h_out = (hp - d_h * (kh - 1) - 1) // sh + 1
    w_out = (wp - d_w * (kw - 1) - 1) // sw + 1

    out = torch.zeros(n, g, h_out, w_out, dtype=input.dtype, device=input.device)
    for ki in range(kh):
        rows = torch.arange(h_out, device=input.device, dtype=torch.long) * sh + ki * d_h
        for kj in range(kw):
            cols = torch.arange(w_out, device=input.device, dtype=torch.long) * sw + kj * d_w
            sub = x_pad[:, :, rows[:, None], cols[None, :]]
            wgk = weight[:, 0, ki, kj].reshape(1, g, 1, 1)
            out = out + sub * wgk

    if bias is not None:
        out = out + bias.reshape(1, g, 1, 1)
    return out


@register_decomposition(aten.convolution.default)
def decompose_convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: Union[int, torch.SymInt],
):
    """
    1. Pointwise 1x1 on spatial 1x1 (groups==1): rewrite to F.linear so backends
       that struggle with tiny spatial convs (e.g. SE after AdaptiveAvgPool2d(1)) see
       aten.mm / linear lowering instead.

    2. Grouped conv when Cin//groups == Cout//groups == 1: _group_conv_cin1_cout1.

    Otherwise returns NotImplemented (Inductor uses the default aten.convolution).

    Note
    ----
    The grouped path is not performance-optimized; it exists for correctness experiments.
    """
    try:
        gcount = operator.index(groups)
    except (TypeError, ValueError):
        return NotImplemented

    if _can_rewrite_pointwise_conv_on_1x1_spatial_to_linear(
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        gcount,
    ):
        return _apply_pointwise_conv_on_1x1_spatial_as_linear(input, weight, bias)

    # groups==1, non-1x1 spatial: keep default aten.convolution (plain conv).
    if gcount == 1:
        return NotImplemented

    cin = input.shape[1]
    cout = weight.shape[0]
    cin_pg = cin // gcount
    cout_pg = cout // gcount
    supported = (
        not transposed
        and cin % gcount == 0
        and cout % gcount == 0
        and cin_pg == 1
        and cout_pg == 1
        and weight.shape[1] == 1
    )
    if not supported:
        raise NotImplementedError(
            "PyTorchSim aten.convolution decomposition supports grouped conv only when "
            "Cin//groups == 1 and Cout//groups == 1 (i.e. per-group Cin and Cout are 1). "
            "For general group convolution, use the default kernel or a dedicated template kernel."
        )
    return _group_conv_cin1_cout1(
        input,
        weight,
        bias,
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        gcount,
    )

@register_decomposition(aten._native_multi_head_attention.default)
def decompose_native_multi_head_attention(
    query,
    key,
    value,
    embed_dim: int,
    num_heads: int,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights: bool = False,
):
    """
    Decompose _native_multi_head_attention into scaled_dot_product_attention operations.

    Based on F.scaled_dot_product_attention and nn.MultiheadAttention implementation:
    1. QKV projection (if needed - but query/key/value may already be projected)
    2. Reshape to multi-head format
    3. Scaled dot product: Q @ K^T / sqrt(head_dim)
    4. Softmax
    5. Attention @ V
    6. Reshape back and output projection
    """
    head_dim = embed_dim // num_heads
    scale_factor = 1.0 / math.sqrt(head_dim)

    # Get input shapes - assuming [batch, seq_len, embed_dim] format
    query_shape = query.shape
    if len(query_shape) == 3:
        # [batch, seq_len, embed_dim] format
        batch_size = query_shape[0]
        seq_len = query_shape[1]
    elif len(query_shape) == 2:
        # [seq_len, embed_dim] -> add batch dimension
        batch_size = 1
        seq_len = query_shape[0]
        query = query.unsqueeze(0)  # [1, seq_len, embed_dim]
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    else:
        # Fallback: assume first dim is batch, second is seq_len
        batch_size = query_shape[0] if len(query_shape) > 0 else 1
        seq_len = query_shape[1] if len(query_shape) > 1 else query_shape[0]

    # Step 1: QKV projection (if query/key/value are not already projected)
    # In many cases, query/key/value are already projected, so we check if qkv_weight is used
    # For now, assume they might need projection
    # Note: In practice, _native_multi_head_attention often receives already projected inputs

    # Reshape for projection: [batch, seq_len, embed_dim] -> [batch*seq_len, embed_dim]
    if len(query.shape) == 3:
        query_flat = query.view(-1, embed_dim)
        key_flat = key.view(-1, embed_dim)
        value_flat = value.view(-1, embed_dim)
    else:
        query_flat = query
        key_flat = key
        value_flat = value

    # QKV projection using qkv_weight and qkv_bias
    # Check if GQA (Grouped Query Attention) is used
    # Standard MHA: qkv_weight shape = [3*embed_dim, embed_dim]
    # GQA: qkv_weight shape = [embed_dim + 2*kv_embed_dim, embed_dim] where kv_embed_dim < embed_dim
    qkv_weight_total = qkv_weight.shape[0]

    # Determine if GQA: if qkv_weight is not exactly 3*embed_dim, it might be GQA
    if qkv_weight_total == 3 * embed_dim:
        # Standard MHA: split equally
        qkv_weight_q, qkv_weight_k, qkv_weight_v = torch.split(qkv_weight, embed_dim, dim=0)
        if qkv_bias is not None:
            qkv_bias_q, qkv_bias_k, qkv_bias_v = torch.split(qkv_bias, embed_dim, dim=0)
        else:
            qkv_bias_q = qkv_bias_k = qkv_bias_v = None
        kv_embed_dim = embed_dim
        kv_heads = num_heads
    else:
        # GQA: Q has embed_dim, K and V share the rest
        # Assume Q = embed_dim, K = V = (qkv_weight_total - embed_dim) / 2
        q_dim = embed_dim
        kv_dim = (qkv_weight_total - embed_dim) // 2
        qkv_weight_q = qkv_weight[:q_dim]
        qkv_weight_k = qkv_weight[q_dim:q_dim + kv_dim]
        qkv_weight_v = qkv_weight[q_dim + kv_dim:]
        if qkv_bias is not None:
            qkv_bias_q = qkv_bias[:q_dim]
            qkv_bias_k = qkv_bias[q_dim:q_dim + kv_dim]
            qkv_bias_v = qkv_bias[q_dim + kv_dim:]
        else:
            qkv_bias_q = qkv_bias_k = qkv_bias_v = None
        kv_embed_dim = kv_dim
        kv_heads = kv_embed_dim // head_dim  # Number of KV heads

    # Project Q, K, V
    q = torch.nn.functional.linear(query_flat, qkv_weight_q, qkv_bias_q)
    k = torch.nn.functional.linear(key_flat, qkv_weight_k, qkv_bias_k)
    v = torch.nn.functional.linear(value_flat, qkv_weight_v, qkv_bias_v)

    # Reshape back: [batch*seq_len, embed_dim] -> [batch, seq_len, embed_dim]
    q = q.view(batch_size, seq_len, embed_dim)
    k = k.view(batch_size, seq_len, kv_embed_dim)
    v = v.view(batch_size, seq_len, kv_embed_dim)

    # Step 2: Reshape to multi-head format
    # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, kv_heads, head_dim)
    v = v.view(batch_size, seq_len, kv_heads, head_dim)

    # Transpose to [batch, num_heads, seq_len, head_dim] for bmm
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)  # [batch, kv_heads, seq_len, head_dim]
    v = v.transpose(1, 2)  # [batch, kv_heads, seq_len, head_dim]

    # GQA: If key/value have fewer heads, repeat them to match query heads
    if kv_heads < num_heads:
        repeat_factor = num_heads // kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)  # [batch, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(repeat_factor, dim=1)  # [batch, num_heads, seq_len, head_dim]

    # Step 3: Scaled dot product attention
    # Scale Q
    q_scaled = q * scale_factor

    # Q @ K^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    k_transposed = k.transpose(-2, -1)  # [batch, num_heads, head_dim, seq_len]
    scores = torch.matmul(q_scaled, k_transposed)  # [batch, num_heads, seq_len, seq_len]

    # Step 4: Apply mask if provided
    if mask is not None:
        attn_bias = torch.zeros_like(scores)
        if mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_bias + mask
        scores = scores + attn_bias

    # Step 5: Softmax along the last dimension (seq_len dimension)
    attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

    # Step 6: Attention @ V
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    # -> [batch, num_heads, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, v)

    # Step 7: Reshape back to [batch, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

    # Step 8: Output projection
    attn_output_flat = attn_output.view(-1, embed_dim)
    output = torch.nn.functional.linear(attn_output_flat, proj_weight, proj_bias)
    output = output.view(batch_size, seq_len, embed_dim)

    if need_weights:
        # Return attention weights: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
        attn_weights_mean = attn_weights.mean(dim=1)  # Average over heads
        return output, attn_weights_mean
    else:
        return (output, None)

@register_decomposition(aten.frexp.Tensor)
def decompose_frexp(x: torch.Tensor):
    """Split ``x``into a mantissa in [0.5, 1) and an integer exponent. 
    
    ``ops.frexp`` cannot be implemented in ``mlir_ops.py``: the CSE proxy in
    ``mlir_common.py`` unpacks exactly ``(code, ret_info)`` from every op and
    hands back a single CSE variable, while Inductor's ``register_frexp``
    subscripts the result as ``ops.frexp(x)[0]`` / ``[1]``. Multi-output ops are
    handled a level up instead -- ``aten.sort`` does the same thing via a custom
    lowering. frexp needs no template, so a decomposition into ops that already
    exist is enough.

    The obvious ``floor(log2|x|) + 1`` formulation is *not* usable here. The
    simulated ``log2``is not exact on powers of two (measured: 16 of 164
    mismatches over 2^-20..2^20, up to 9.5e-07), so ``floor`` slips by one and
    the mantissa lands just under 1.0 instead of at 0.5. Comparing against
    ``finfo.tiny`` is broken too -- subnormal operands compare as if flushed, so
    a float-side subnormal test never fires.

    Bit surgery avoids both problems and is exact. For a normal float32
    ``x = (-1)^s * 1.mant * 2^(expf - 127)``, so forcing the biased exponent to
    126 yields ``m = (-1)^s * 0.1mant``in [0.5, 1) and leaves ``e = expf - 126``.
    Subnormals are first scaled into the normal ranges by 2**24 and the 24 is
    taken back off the exponent. Zero, the infinities and NaN pass through with
    an exponent of 0, matching ``torch.frexp``.

    Verified against ``torch.frexp`` on the npu backend across normals, powers
    of two, +/-0, subnormals down to 1.4e-45, +/-inf and NaN: mantissa and
    exponent both match exactly.
    """
    # The masks below are float32 layouts; let Inductor handle anything else.
    if x.dtype != torch.float32:
        return NotImplemented
    
    bits = x.view(torch.int32)
    abs_bits = bits & 0x7FFFFFFF
    exp_field = (bits >> 23) & 0xFF

    is_zero = abs_bits == 0
    is_inf_nan = exp_field == 255
    # Subnormals must be detected on the integer side: the float comparison
    # against finfo.tiny reports false for every subnormal on this target.
    is_subnormal = (exp_field == 0) & (abs_bits != 0)

    scaled = torch.where(is_subnormal, x * 16777216.0, x)   # 2**24
    scaled_bits = scaled.view(torch.int32)

    # Keep sign + mantissa, overwrite the exponent with 126 (i.e. 2**-1).
    mantissa = ((scaled_bits & 0x807FFFFF) | 0x3F000000).view(torch.float32)
    exponent = ((scaled_bits >> 23) & 0xFF) - 126
    exponent = exponent - torch.where(
        is_subnormal, torch.full_like(exponent, 24), torch.zeros_like(exponent)
    )

    passthrough = is_zero | is_inf_nan
    mantissa = torch.where(passthrough, x, mantissa)
    exponent = torch.where(passthrough, torch.zeros_like(exponent), exponent)
    return mantissa, exponent

# Lower roll as narrow + cat, then REALIZE: torch's decomposition is a modular gather the
# affine-only DMA cannot express, and even narrow+cat fuses into a modular reshape.
# copy_input forces the buffer boundary .contiguous() does not.
from torch._inductor.decomposition import decompositions as _inductor_decompositions
from torch._inductor import lowering as _ind_lowering
from torch._inductor import ir as _ind_ir

_inductor_decompositions.pop(aten.roll.default, None)


def _roll_lowering(x, shifts, dims=()):
    if not isinstance(shifts, (list, tuple)):
        shifts = (shifts,)
    if not isinstance(dims, (list, tuple)):
        dims = (dims,)
    slice_l = _ind_lowering.lowerings[aten.slice.Tensor]
    cat_l = _ind_lowering.lowerings[aten.cat.default]
    view_l = _ind_lowering.lowerings[aten.view.default]

    # Realize the INPUT: roll's producer is often a reshaped view (e.g. swinv2's window_reverse),
    # and our slice's offset would otherwise fuse into that reshape's modular index. Reading a
    # materialized buffer makes each slice a plain contiguous read.
    x = _ind_ir.ExternKernel.copy_input(x)

    def roll_dim(t, shift, dim):
        n = int(t.get_size()[dim])
        s = int(shift) % n
        if s == 0:
            return t
        front = slice_l(t, dim, n - s, n)          # narrow(t, dim, n-s, s)
        back = slice_l(t, dim, 0, n - s)
        return cat_l([front, back], dim)

    if len(dims) == 0:
        numel = 1
        for d in x.get_size():
            numel *= int(d)
        result = view_l(roll_dim(view_l(x, [numel]), shifts[0], 0), list(x.get_size()))
    else:
        result = x
        for shift, dim in zip(shifts, dims):
            result = roll_dim(result, shift, dim)
    return _ind_ir.ExternKernel.copy_input(result)


_ind_lowering.lowerings[aten.roll.default] = _roll_lowering