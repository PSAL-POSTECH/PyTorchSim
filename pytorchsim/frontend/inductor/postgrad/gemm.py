"""The matmul family: a row times a matrix as vector work, matmuls with nothing to compute, and attention
bmms against keys or values repeated per query head."""

import torch
from torch import fx
from torch._inductor.pattern_matcher import Arg, CallFunction, CallFunctionVarArgs

from pytorchsim.frontend.inductor.postgrad import fake_value, register_rewrite

aten = torch.ops.aten


# mm_row, addmm_row: a 1 x K row times a matrix

def _is_row(match):
    """Real floating operands, the left a 1 x K row: an mm or a plain addmm (no beta or alpha)."""
    a, b = match.args[-2:]
    av, bv = fake_value(a), fake_value(b)
    return (av is not None and bv is not None and av.dim() == 2 and bv.dim() == 2
            and av.shape[0] == 1 and av.is_floating_point() and bv.is_floating_point())


def _row_times(a, b):
    """The row stored first (a no-op when it already is), broadcast down the matrix, a multiply and a sum over
    K: every block reads the stored row, where a row computed in the kernel is spread over the lanes per block."""
    from torch._inductor import inductor_prims
    return (inductor_prims.force_stride_order(a, [a.shape[1], 1]).reshape(-1, 1) * b).sum(0, keepdim=True)


@register_rewrite(CallFunction(aten.mm.default, Arg(), Arg()), _is_row)
def mm_row(node, a, b):
    """A 1 x K row times a K x N matrix as _row_times: vector work, where the mm template would run it on
    the array."""
    return _row_times, [a, b]


@register_rewrite(CallFunction(aten.addmm.default, Arg(), Arg(), Arg()), _is_row)
def addmm_row(node, bias, a, b):
    """mm_row plus the bias."""
    return (lambda bias, a, b: _row_times(a, b) + bias), [bias, a, b]


# gemm_zero_length: matmuls with nothing to compute

def _is_zero_length(bias):
    """An mm-family node whose output is empty, or whose K is zero with no bias to add."""
    def check(match):
        node = match.output_node()
        out, a = fake_value(node), fake_value(node.args[bias])
        if out is None or a is None or not all(isinstance(s, int) for s in (*out.shape, *a.shape)):
            return False
        return out.numel() == 0 or (not bias and a.shape[-1] == 0)
    return check


def gemm_zero_length(node, *args, **kwargs):
    """A matmul with a zero-length M or N, or K and no bias, as zeros of its output: nothing to compute,
    where the mm template would be asked for an empty tile."""
    out = fake_value(node)
    shape, dtype = list(out.shape), out.dtype
    first = next(a for a in args if isinstance(a, fx.Node))
    return (lambda t: t.new_zeros(shape, dtype=dtype)), [first]


for _op, _bias in ((aten.mm.default, 0), (aten.bmm.default, 0), (aten.addmm.default, 1),
                   (aten.baddbmm.default, 1)):
    register_rewrite(CallFunctionVarArgs(_op), _is_zero_length(_bias))(gemm_zero_length)


# gqa_shared_kv: an attention bmm against keys or values repeated per query head (repeat_kv)

_VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)


def _swapped_last_two(node):
    """Whether a permute or transpose node swaps its last two dims (True), keeps every dim (False), else None."""
    n = fake_value(node.args[0]).dim()
    if node.target is aten.permute.default:
        dims = [d % n for d in node.args[1]]
    else:
        dims = list(range(n))
        i, j = node.args[1] % n, node.args[2] % n
        dims[i], dims[j] = dims[j], dims[i]
    if dims == list(range(n)):
        return False
    return True if dims == list(range(n - 2)) + [n - 1, n - 2] else None


def _repeated_kv(node):
    """(kv, R, transposed) when `node` is views of repeat_kv's copy of kv [B, H, S, D]: unsqueeze at 2,
    expand to [B, H, R, S, D] (R > 1) and clone, then views and last-two-dim swaps (transposed if odd)."""
    swaps = 0
    while isinstance(node, fx.Node) and node.target is not aten.clone.default:
        if node.target is aten.expand.default:
            if list(fake_value(node).shape) != list(fake_value(node.args[0]).shape):
                return None
        elif node.target in (aten.permute.default, aten.transpose.int):
            swapped = _swapped_last_two(node)
            if swapped is None:
                return None
            swaps += swapped
        elif node.target not in _VIEWS:
            return None
        node = node.args[0]
    expand = node.args[0] if isinstance(node, fx.Node) else None
    unsqueeze = expand.args[0] if isinstance(expand, fx.Node) and expand.target is aten.expand.default else None
    if not (isinstance(unsqueeze, fx.Node) and unsqueeze.target is aten.unsqueeze.default
            and unsqueeze.args[1] == 2):
        return None
    kv = unsqueeze.args[0]
    kvv, ev = fake_value(kv), fake_value(expand)
    if kvv is None or ev is None or kvv.dim() != 4 or ev.dim() != 5:
        return None
    if not all(isinstance(s, int) for s in ev.shape):
        return None
    B, H, S, D = kvv.shape
    R = ev.shape[2]
    if list(ev.shape) != [B, H, R, S, D] or R <= 1:
        return None
    return kv, R, bool(swaps % 2)


def _bmm_repeated_kv(b):
    """_repeated_kv of a bmm's right operand, when it is the [B*H*R, S, D] (or transposed) batch of it."""
    got = _repeated_kv(b)
    bv = fake_value(b)
    if got is None or bv is None:
        return None
    kv, R, transposed = got
    B, H, S, D = fake_value(kv).shape
    return got if list(bv.shape) == ([B * H * R, D, S] if transposed else [B * H * R, S, D]) else None


def _is_gqa(match):
    """A bmm whose right operand is a repeat_kv copy, and a left operand of the same batch."""
    a, b = match.args[:2]
    av, bv = fake_value(a), fake_value(b)
    return (av is not None and bv is not None and av.dim() == 3 and isinstance(av.shape[1], int)
            and _bmm_repeated_kv(b) is not None and av.shape[0] == bv.shape[0])


@register_rewrite(CallFunction(aten.bmm.default, Arg(), Arg()), _is_gqa)
def gqa_shared_kv(node, a, b):
    """The R query heads that share a key or value head as one bmm over B*H batches of R*M rows against
    that head itself: the same products, without the R copies repeat_kv writes."""
    kv, r, transposed = _bmm_repeated_kv(b)
    B, H, S, D = fake_value(kv).shape
    m, k = fake_value(a).shape[1:]
    out = list(fake_value(node).shape)

    def shared(a, kv):
        right = kv.reshape(B * H, S, D)
        right = right.transpose(1, 2) if transposed else right
        return torch.bmm(a.reshape(B * H, r * m, k), right).reshape(out)
    return shared, [a, kv]


# gqa_shared_kv_fused: the fused SDPA handed repeat_kv's copies, handed the shared heads instead

FUSED_SDPA = "torchsim::sdpa_fused"


def gqa_shared_kv_fused(graph):
    """Each fused SDPA whose key and value are repeat_kv copies [B, H*R, S, D] takes the shared [B, H, S, D]
    ones: its kernels group query heads by Hq // H. Returns how many it rewired."""
    from pytorchsim.frontend.inductor.postgrad import count
    changed = 0
    for node in graph.nodes:
        name = getattr(node.target, "name", None)
        if node.op != "call_function" or not callable(name) or name() != FUSED_SDPA:
            continue
        q, k, v = node.args[:3]
        got = [_repeated_kv(t) for t in (k, v)]
        if any(g is None or g[2] for g in got) or got[0][1] != got[1][1]:
            continue
        if any(list(fake_value(t).shape) != [*fake_value(g[0]).shape[:1], fake_value(g[0]).shape[1] * g[1],
                                             *fake_value(g[0]).shape[2:]] for t, g in zip((k, v), got)):
            continue
        if fake_value(q).shape[1] != fake_value(k).shape[1]:
            continue
        node.args = (q, got[0][0], got[1][0], *node.args[3:])
        count("gqa_shared_kv_fused", "rewritten")
        changed += 1
    return changed
