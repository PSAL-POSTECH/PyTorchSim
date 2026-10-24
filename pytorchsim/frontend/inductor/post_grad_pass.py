"""The npu's post-grad pass (Inductor's device_custom_pass, run on npu graphs only): rewrites that
keep ops with no npu kernel on the device. The one-op rewrites, Inductor graph patterns, run first:

    histc                    aten.histc counted on the device
    mm_row, addmm_row        a 1 x K row times a matrix as a multiply and a sum: vector work
    convolution_transposed   a transposed conv as its direct equivalent
    convolution_1d           a 1-D conv as a 2-D one over a height of 1
    gemm_zero_length         a matmul with an empty output, or K = 0 and no bias, as zeros

then complex_real_pairs: complex values as real [..., 2] pairs, by each op's rule in _REAL_PAIR.
A rewrite returns plain torch code for the node's place, or a Decline that leaves it (logged,
counted in CALLS); how to add one is docs/post_grad_rewrite.md. Ops with a Triton kernel of their
own are dispatcher/.
"""

import contextlib
import functools

import torch
from torch import fx
from torch.fx import traceback as fx_traceback
from torch._inductor.custom_graph_pass import CustomGraphModulePass, get_hash_for_files
from torch._inductor.pattern_matcher import (Arg, CallFunction, CallFunctionVarArgs,
                                             PatternMatcherPass, register_graph_pattern)

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline

aten = torch.ops.aten

logger = config.setup_logger()

# The pass: its graph patterns, then the complex values

PATTERNS = PatternMatcherPass(pass_name="pytorchsim_post_grad_pass")

#: rewrite name -> {"rewritten": n, "fallback": n}: the nodes each rewrite replaced or left.
CALLS = {}


class _PostGradPass(CustomGraphModulePass):
    """The graph patterns, then complex values as real pairs; its uuid hashes this file, so an edit here
    is a new cache key."""

    def __call__(self, gm: fx.GraphModule) -> None:
        """Replacements trace under the graph's fake mode: Inductor sets it, a caller outside it may not."""
        from torch._inductor.virtualized import V

        fake_mode = _fake_mode(gm.graph)
        with V.set_fake_mode(fake_mode) if fake_mode is not None else contextlib.nullcontext():
            changed = PATTERNS.apply(gm) + complex_real_pairs(gm.graph, fake_mode)
        if changed:
            gm.graph.eliminate_dead_code()
            gm.recompile()

    def uuid(self):
        return get_hash_for_files((__file__,))


pytorchsim_post_grad_pass = _PostGradPass()


# register_rewrite: the form every one-op rewrite takes, its fallback and its counts

def register_rewrite(pattern, extra_check=lambda match: True):
    """Register fn(node, *args, **kwargs) for `pattern` (`extra_check` picks the nodes it applies to). fn
    returns (replacement, args): plain torch code Inductor traces to take the node's place; or a Decline:
    the node stays, logged with where it runs instead (the fallback). Each counts in CALLS."""
    def register(fn):
        def handler(match, *args, **kwargs):
            node = match.output_node()
            got = fn(node, *args, **kwargs)
            if isinstance(got, Decline):
                _count(fn.__name__, "fallback")
                logger.info("[post_grad_pass] %s stays: %s; it runs as %s", node.target, got,
                            fallback_route(node.target))
                return
            replacement, example = got
            match.replace_by_example(replacement, example)
            _count(fn.__name__, "rewritten")
        register_graph_pattern(pattern, extra_check=extra_check, pass_dict=PATTERNS)(handler)
        return fn
    return register


def fallback_route(target):
    """Where a node left as `target` runs: Inductor's own kernel, or an extern call aten makes on the
    CPU, outside the simulator, when Inductor has no lowering for it or falls back."""
    from torch._inductor.lowering import fallbacks, lowerings
    packet = getattr(target, "overloadpacket", None)
    lowered = target in lowerings or packet in lowerings
    if lowered and target not in fallbacks and packet not in fallbacks:
        return "Inductor's own kernel"
    return "an extern aten call on the CPU (not simulated)"


def _count(rewrite, route):
    counts = CALLS.setdefault(rewrite, {})
    counts[route] = counts.get(route, 0) + 1


# histc: aten.histc counted on the device

@register_rewrite(CallFunctionVarArgs(aten.histc.default))
def histc(node, x, *rest, **kwargs):
    """aten.histc counted on the device (counts)."""

    def counts(x, *, bins, lo, hi):
        """aten's histc: bin (x - lo) * bins / (hi - lo), integral in int64 and floating in x's dtype, the top
        edge in the last bin, outside [lo, hi] dropped; lo == hi takes x's range, widened by 1 if one value."""
        integral = not x.is_floating_point()
        flat = x.reshape(-1).to(torch.int64 if integral else x.dtype)
        if lo == hi and x.numel():
            lo, hi = flat.amin(0), flat.amax(0)
            one = lo == hi
            lo, hi = torch.where(one, lo - 1, lo), torch.where(one, hi + 1, hi)
        else:
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            if integral:
                lo, hi = int(lo), int(hi)
            if hi <= lo:
                raise RuntimeError(f"torch.histc: max must be larger than min (min {lo}, max {hi})")
        scaled = (flat - lo) * bins
        if integral:
            pos = torch.div(scaled, hi - lo, rounding_mode="floor")
        else:
            pos = (scaled / (hi - lo)).to(torch.int64)
        pos = pos.clamp_max(bins - 1)
        inside = (flat >= lo) & (flat <= hi)
        seats = torch.arange(bins, dtype=torch.int64, device=x.device)
        hits = (pos.reshape(-1, 1) == seats.reshape(1, -1)) & inside.reshape(-1, 1)
        return hits.sum(0).to(x.dtype)

    if _val(x) is None:
        return Decline("its input has no fake value")
    said = dict(zip(("bins", "min", "max"), rest), **kwargs)
    return functools.partial(counts, bins=int(said.get("bins", 100)), lo=said.get("min", 0),
                             hi=said.get("max", 0)), [x]


# mm_row, addmm_row: a 1 x K row times a matrix

def _is_row(match):
    """Real floating operands, the left a 1 x K row: an mm or a plain addmm (no beta or alpha)."""
    a, b = match.args[-2:]
    av, bv = _val(a), _val(b)
    return (av is not None and bv is not None and av.dim() == 2 and bv.dim() == 2
            and av.shape[0] == 1 and av.is_floating_point() and bv.is_floating_point())


@register_rewrite(CallFunction(aten.mm.default, Arg(), Arg()), _is_row)
def mm_row(node, a, b):
    """A 1 x K row times a K x N matrix as the row broadcast down the matrix, a multiply and a sum over K:
    vector work, where the mm template would run it on the array."""
    return (lambda a, b: (a.reshape(-1, 1) * b).sum(0, keepdim=True)), [a, b]


@register_rewrite(CallFunction(aten.addmm.default, Arg(), Arg(), Arg()), _is_row)
def addmm_row(node, bias, a, b):
    """mm_row plus the bias."""
    return (lambda bias, a, b: (a.reshape(-1, 1) * b).sum(0, keepdim=True) + bias), [bias, a, b]


# convolution_transposed, convolution_1d: convolutions the conv template takes once rewritten

@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))),
             lambda match: match.args[6] is True)
def convolution_transposed(node, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    """A transposed convolution as its direct equivalent: the input spread by the stride and padded by
    dilation * (kernel - 1) - padding (plus output_padding after), the weight's in and out channels
    swapped per group, flipped, and spread by the dilation; then stride, dilation 1, no padding."""

    def per_dim(seq, n):
        seq = list(seq)
        return seq * n if len(seq) == 1 else seq

    def spread(t, lead, extents, factor):
        """factor - 1 zeros between neighbouring elements along each of the trailing dims."""
        if all(f == 1 for f in factor):
            return t
        n = len(extents)
        paired = list(lead)
        for extent in extents:
            paired += [extent, 1]
        pad = []
        for i in reversed(range(n)):
            pad += [0, factor[i] - 1, 0, 0]
        y = torch.constant_pad_nd(t.reshape(paired), pad, 0.0)
        y = y.reshape(list(lead) + [extents[i] * factor[i] for i in range(n)])
        for i in range(n):
            if factor[i] != 1:
                y = y.narrow(len(lead) + i, 0, extents[i] * factor[i] - (factor[i] - 1))
        return y

    def direct(x, weight, bias=None):
        batchless = x.dim() == weight.dim() - 1
        if batchless:
            x = x.unsqueeze(0)
        batch, in_chan, *spatial = x.shape
        _, per_group, *kernel = weight.shape
        n = len(kernel)
        out_chan = per_group * groups
        st, pd, dl, op = (per_dim(v, n) for v in (stride, padding, dilation, output_padding))
        y = spread(x, [batch, in_chan], spatial, st)
        pad = []
        for i in reversed(range(n)):
            lo = dl[i] * (kernel[i] - 1) - pd[i]
            pad += [lo, lo + op[i]]
        if any(p != 0 for p in pad):
            y = torch.constant_pad_nd(y, pad, 0.0)
        w = weight.reshape(groups, in_chan // groups, per_group, *kernel).transpose(1, 2)
        w = w.reshape(out_chan, in_chan // groups, *kernel)
        if any(k != 1 for k in kernel):
            w = w.flip(list(range(2, 2 + n)))
        w = spread(w, [out_chan, in_chan // groups], kernel, dl)
        if n == 1:
            out = _conv1d_as_conv2d(y, w, bias, [1], [0], [1], groups)
        else:
            out = torch.convolution(y, w, bias, [1] * n, [0] * n, [1] * n, False, [0] * n, groups)
        return out.squeeze(0) if batchless else out

    return direct, [x, weight] + ([bias] if bias is not None else [])


@register_rewrite(CallFunction(aten.convolution.default, *(Arg() for _ in range(9))),
                  lambda match: match.args[6] is False and len(match.args[3]) == 1)
def convolution_1d(node, x, weight, bias, stride, padding, dilation, transposed, output_padding,
                   groups):
    """A 1-D convolution as a 2-D one over a height of 1, which the conv template takes; its backward
    graph holds convolution_backward either way, so this needs no place before AOTAutograd."""
    def as_2d(x, weight, bias=None):
        return _conv1d_as_conv2d(x, weight, bias, stride, padding, dilation, groups)

    return as_2d, [x, weight] + ([bias] if bias is not None else [])


def _conv1d_as_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """conv1d of [N, C, L] as conv2d of [N, C, 1, L]: the height takes stride and dilation 1, no padding."""
    out = torch.convolution(x.unsqueeze(2), weight.unsqueeze(2), bias, [1, stride[0]],
                            [0, padding[0]], [1, dilation[0]], False, [0, 0], groups)
    return out.squeeze(2)


# gemm_zero_length: matmuls with nothing to compute

def _is_zero_length(bias):
    """An mm-family node whose output is empty, or whose K is zero with no bias to add."""
    def check(match):
        node = match.output_node()
        out, a = _val(node), _val(node.args[bias])
        if out is None or a is None or not all(isinstance(s, int) for s in (*out.shape, *a.shape)):
            return False
        return out.numel() == 0 or (not bias and a.shape[-1] == 0)
    return check


def gemm_zero_length(node, *args, **kwargs):
    """A matmul with a zero-length M or N, or K and no bias, as zeros of its output: nothing to compute,
    where the mm template would be asked for an empty tile."""
    out = _val(node)
    shape, dtype = list(out.shape), out.dtype
    first = next(a for a in args if isinstance(a, fx.Node))
    return (lambda t: t.new_zeros(shape, dtype=dtype)), [first]


for _op, _bias in ((aten.mm.default, 0), (aten.bmm.default, 0), (aten.addmm.default, 1),
                   (aten.baddbmm.default, 1)):
    register_rewrite(CallFunctionVarArgs(_op), _is_zero_length(_bias))(gemm_zero_length)


# complex_real_pairs: complex values as real pairs, one rule per complex op in _REAL_PAIR

def complex_real_pairs(graph, fake_mode) -> int:
    """Each component of complex values whose ops _REAL_PAIR has a rule for becomes real [..., 2] pairs;
    any other component stays, logged with why and where it runs. Returns the components rewritten."""

    def components(nodes):
        """The complex nodes grouped by the complex values joining them, each group in graph order."""
        parent = {n: n for n in nodes}

        def root(n):
            while parent[n] is not n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for n in nodes:
            for a in n.all_input_nodes:
                if a in parent:
                    parent[root(a)] = root(n)
        groups = {}
        for n in nodes:
            groups.setdefault(root(n), []).append(n)
        return list(groups.values())

    def why_not_real(component):
        """Why the component has no real-pair form, or None: a node _REAL_PAIR has no rule for (a placeholder or
        attribute among them), or a complex value read by anything but _REAL_PAIR and view_as_real (the output too)."""
        for n in component:
            if n.target not in _REAL_PAIR:
                return f"{n.op} {n.target}"
            for user in n.users:
                if user.target is not aten.view_as_real.default and user.target not in _REAL_PAIR:
                    return f"{user.op} {user.name} reads {n.name}'s complex value"
        return None

    def to_real_pairs(graph, fake_mode, component):
        """Rewrite one eligible component: each complex value becomes its [..., 2] pair and every view_as_real of
        it reads the pair."""

        def inserter(graph, fake_mode, origin):
            """call(target, *args, **kwargs): a node just before origin, made through FX's tracer under origin's meta
            so FX copies its provenance and records origin in from_node; its fake value is set, as nothing after does."""
            def unwrap(a):
                if isinstance(a, (list, tuple)):
                    return type(a)(unwrap(x) for x in a)
                if isinstance(a, dict):
                    return {k: unwrap(v) for k, v in a.items()}
                return _val(a)

            tracer = fx.proxy.GraphAppendingTracer(graph)

            def call(target, *args, **kwargs):
                with graph.inserting_before(origin), fx_traceback.preserve_node_meta(), \
                        fx_traceback.set_current_meta(origin):
                    node = tracer.create_node("call_function", target, args, kwargs)
                if fake_mode is not None:
                    with fake_mode:
                        node.meta["val"] = target(*unwrap(args), **unwrap(kwargs))
                return node

            return call

        pair = {}
        for n in component:
            pair[n] = _REAL_PAIR[n.target](inserter(graph, fake_mode, n), n, pair)

        for n in component:
            for user in list(n.users):
                if user.target is aten.view_as_real.default:
                    user.replace_all_uses_with(pair[n])

    changed = 0
    for component in components([n for n in graph.nodes if _is_complex(n)]):
        why = why_not_real(component)
        if why is not None:
            _count("complex_real_pairs", "fallback")
            logger.info("[post_grad_pass] %d complex nodes stay: %s; complex ops run as %s",
                        len(component), why, fallback_route(component[0].target))
            continue
        to_real_pairs(graph, fake_mode, component)
        _count("complex_real_pairs", "rewritten")
        changed += 1
    return changed


def _pair_of_view(call, n, pair):
    """view_as_complex(x): its pair is x."""
    return n.args[0]


def _pair_of_unsqueeze(call, n, pair):
    """unsqueeze(z, dim): the pair unsqueezed before its last axis."""
    dim = n.args[1]
    return call(aten.unsqueeze.default, pair[n.args[0]], dim if dim >= 0 else dim - 1)


def _pair_of_mul(call, n, pair):
    """z * w as (zr*wr - zi*wi, zr*wi + zi*wr) when w is complex (a value or a Python complex); z * r scales
    the pair when r is real."""
    a, b = n.args[0], n.args[1]
    cx, other = (a, b) if _is_complex(a) else (b, a)
    if not (_is_complex(other) or isinstance(other, complex)):
        if isinstance(other, fx.Node):
            other = call(aten.unsqueeze.default, other, -1)
        return call(aten.mul.Tensor, pair[cx], other)
    ar, ai = call(aten.select.int, pair[cx], -1, 0), call(aten.select.int, pair[cx], -1, 1)
    if isinstance(other, complex):
        br, bi = other.real, other.imag
    else:
        br, bi = call(aten.select.int, pair[other], -1, 0), call(aten.select.int, pair[other], -1, 1)
    re = call(aten.sub.Tensor, call(aten.mul.Tensor, ar, br), call(aten.mul.Tensor, ai, bi))
    im = call(aten.add.Tensor, call(aten.mul.Tensor, ar, bi), call(aten.mul.Tensor, ai, br))
    return _pair(call, re, im)


def _pair_of_polar(call, n, pair):
    """polar(abs, angle): its pair is (abs*cos(angle), abs*sin(angle))."""
    r, angle = n.args[:2]
    return _pair(call, call(aten.mul.Tensor, r, call(aten.cos.default, angle)),
                 call(aten.mul.Tensor, r, call(aten.sin.default, angle)))


_REAL_PAIR = {
    aten.polar.default: _pair_of_polar,
    aten.view_as_complex.default: _pair_of_view,
    aten.unsqueeze.default: _pair_of_unsqueeze,
    aten.mul.Tensor: _pair_of_mul,
    aten.mul.Scalar: _pair_of_mul,
}


# Shared by the rewrites

def _pair(call, re, im):
    """The [..., 2] real pair of re and im."""
    return call(aten.cat.default, [call(aten.unsqueeze.default, re, -1),
                                   call(aten.unsqueeze.default, im, -1)], -1)


def _val(node):
    """The fake value of a node, or the argument itself when it is not a node."""
    return node.meta.get("val", None) if isinstance(node, fx.Node) else node


def _is_complex(node):
    """Whether the node holds a complex tensor."""
    v = _val(node)
    return isinstance(v, torch.Tensor) and v.is_complex()


def _fake_mode(graph):
    """The fake mode of the graph's tensors, or None."""
    for n in graph.nodes:
        v = _val(n)
        if isinstance(v, torch.Tensor) and hasattr(v, "fake_mode"):
            return v.fake_mode
    return None
