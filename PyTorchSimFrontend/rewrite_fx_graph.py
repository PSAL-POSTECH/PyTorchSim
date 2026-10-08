"""Our rewrites of the FX graph, so ops with no npu kernel stay on the device.

Decompositions (histc, polar) and post-grad passes (grouped_mm, topk, complex to
real pairs), all chained onto Inductor's one post-grad hook by install().
"""

from operator import getitem

import torch
from torch import fx
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.decomposition import register_decomposition

aten = torch.ops.aten
prims = torch.ops.prims


@register_decomposition(aten.histc.default)
def decompose_histc(self, bins: int = 100, min: int = 0, max: int = 0):
    """Counts of an integral tensor, on the device when the bins are the values.

    Exact for bins=N, min=0, max=N-1; any other edges keep the CPU hop.
    """
    if self.is_floating_point():
        return NotImplemented
    if int(min) == 0 and int(max) == int(bins) - 1:
        vals = torch.arange(int(bins), device=self.device, dtype=self.dtype)
        counts = (self.reshape(-1, 1) == vals.reshape(1, -1)).sum(0)
        return counts.to(self.dtype)
    counts = torch.histc(self.to(torch.float32), bins, min, max)
    return counts.to(self.dtype)


@register_decomposition(aten.polar.default)
def decompose_polar(abs, angle):
    """`polar` as a view over a real pair, so it never becomes an extern call.

    The pair is made contiguous because view_as_complex needs a unit last stride.
    """
    pair = torch.stack((abs * torch.cos(angle), abs * torch.sin(angle)), dim=-1)
    return torch.view_as_complex(pair.contiguous())


def is_npu_graph(g) -> bool:
    """Whether any tensor in this graph lives on the npu device."""
    graph = getattr(g, "graph", g)
    for node in graph.nodes:
        val = node.meta.get("val")
        for t in (val if isinstance(val, (tuple, list)) else (val,)):
            if isinstance(t, torch.Tensor) and t.device.type == "npu":
                return True
    return False


def npu_only(fn):
    """The same pass, a no-op on a graph with no npu tensor."""
    def _gated(graph):
        if is_npu_graph(graph):
            fn(graph)
    return _gated


def _identity(fn):
    """A name for a foreign callable that is the same in every process."""
    uuid = getattr(fn, "uuid", None)
    if callable(uuid):
        return uuid()
    module = getattr(fn, "__module__", "?")
    name = getattr(fn, "__qualname__", None)
    return f"{module}.{name}" if name else f"{module}.<unnamed>"


class _Chain(CustomGraphPass):
    """Inductor's single post-grad slot, as a chain that knows what is in it."""

    def __init__(self, foreign):
        self._foreign = foreign
        self._keys = []
        self._passes = []

    def add(self, key, fn) -> bool:
        """Append fn under key, or report that key is already chained."""
        if key in self._keys:
            return False
        self._keys.append(key)
        self._passes.append(fn)
        return True

    def uuid(self):
        head = () if self._foreign is None else (_identity(self._foreign),)
        return head + tuple(self._keys)

    def __call__(self, graph):
        if self._foreign is not None:
            self._foreign(graph)
        for fn in self._passes:
            fn(graph)


def install_once(key, fn):
    """Chain fn onto Inductor's post-grad hook, at most once per key."""
    from torch._inductor import config

    chain = config.post_grad_custom_post_pass
    if not isinstance(chain, _Chain):
        chain = _Chain(chain)
        config.post_grad_custom_post_pass = chain
    return chain.add(key, fn)


def _recompile(g):
    """Drop what a rewrite left dead and refresh the owning module's code."""
    g.eliminate_dead_code()
    if g.owning_module is not None:
        g.owning_module.recompile()


def _grouped_mm_target():
    """transformers' grouped_mm_fallback overload, or None until transformers is imported."""
    ns = getattr(torch.ops, "transformers", None)
    op = getattr(ns, "grouped_mm_fallback", None) if ns is not None else None
    return op.default if op is not None else None


def _emit_dense_grouped_mm(g, inp, weight, offs, s, e, o, dtype, dev):
    """Every expert over the whole input, masked to its own rows and summed.

    Returns the (s, o) result node.
    """
    call = g.call_function
    i64 = torch.int64
    rows = call(prims.iota.default, (s,),
                {"start": 0, "step": 1, "dtype": i64, "device": dev,
                 "requires_grad": False})
    rows2 = call(aten.view.default, (rows, [s, 1]))
    offs64 = call(prims.convert_element_type.default, (offs, i64))
    offs2 = call(aten.view.default, (offs64, [1, e]))
    reached = call(aten.ge.Tensor, (rows2, offs2))
    reached64 = call(prims.convert_element_type.default, (reached, i64))
    expert_of = call(aten.sum.dim_IntList, (reached64, [1]))

    out = call(aten.full.default, ([s, o], 0), {"dtype": dtype, "device": dev})
    for i in range(e):
        wi = call(aten.clone.default, (call(aten.select.int, (weight, 0, i)),),
                  {"memory_format": torch.contiguous_format})
        part = call(aten.mm.default, (inp, wi))
        keep = call(aten.eq.Scalar, (expert_of, i))
        keep2 = call(aten.view.default, (keep, [s, 1]))
        keepf = call(prims.convert_element_type.default, (keep2, dtype))
        out = call(aten.add.Tensor, (out, call(aten.mul.Tensor, (part, keepf))))
    return out


def rewrite_grouped_mm(g) -> None:
    """Replace every `transformers.grouped_mm_fallback` call with the dense form."""
    target = _grouped_mm_target()
    if target is None:
        return
    if isinstance(g, fx.GraphModule):
        g = g.graph
    changed = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not target:
            continue
        inp, weight, offs = node.args[0], node.args[1], node.args[2]
        iv = inp.meta.get("val") if hasattr(inp, "meta") else None
        wv = weight.meta.get("val") if hasattr(weight, "meta") else None
        if iv is None or wv is None or iv.dim() != 2 or wv.dim() != 3:
            continue
        s, e, o = int(iv.shape[0]), int(wv.shape[0]), int(wv.shape[2])
        with g.inserting_before(node):
            dense = _emit_dense_grouped_mm(g, inp, weight, offs, s, e, o,
                                           iv.dtype, iv.device)
        node.replace_all_uses_with(dense)
        changed += 1
    if changed:
        _recompile(g)


def _spent(dtype, largest):
    """The value a taken element is replaced with, so it cannot win again."""
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return info.min if largest else info.max


def _emit_topk(g, x, k, dim, largest, shape, dtype, dev):
    """Emit k rounds of max-and-mask and return the (values, indices) nodes."""
    call = g.call_function
    i64 = torch.int64
    n = int(shape[dim])
    seat_shape = [1] * len(shape)
    seat_shape[dim] = n

    rows = call(prims.iota.default, (n,),
                {"start": 0, "step": 1, "dtype": i64, "device": dev,
                 "requires_grad": False})
    ar = call(aten.view.default, (rows, seat_shape))
    seat = call(aten.expand.default, (ar, list(shape)))
    late = call(aten.full.default, (list(shape), n), {"dtype": i64, "device": dev})
    gone = call(aten.full.default, (list(shape), _spent(dtype, largest)),
                {"dtype": dtype, "device": dev})

    work, vals, idxs = x, [], []
    for _ in range(k):
        best = call(aten.amax.default if largest else aten.amin.default,
                    (work, [dim], True))
        cand = call(aten.where.self, (call(aten.eq.Tensor, (work, best)),
                                      seat, late))
        where = call(aten.amin.default, (cand, [dim], True))
        vals.append(best)
        idxs.append(where)
        work = call(aten.where.self,
                    (call(aten.eq.Tensor, (ar, where)), gone, work))
    return call(aten.cat.default, (vals, dim)), call(aten.cat.default, (idxs, dim))


def rewrite_topk(g) -> None:
    """Replace every `aten.topk.default` whose users are getitem with the rounds."""
    if isinstance(g, fx.GraphModule):
        g = g.graph
    changed = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not aten.topk.default:
            continue
        users = list(node.users)
        if any(u.op != "call_function" or u.target is not getitem for u in users):
            continue
        x = node.args[0]
        xv = x.meta.get("val") if hasattr(x, "meta") else None
        if xv is None or xv.dim() == 0 or not xv.numel():
            continue
        k = int(node.args[1])
        dim = int(node.args[2]) if len(node.args) > 2 else -1
        dim = dim if dim >= 0 else dim + xv.dim()
        largest = bool(node.args[3]) if len(node.args) > 3 else True
        if k < 1 or k > int(xv.shape[dim]):
            continue
        with g.inserting_before(node):
            values, indices = _emit_topk(g, x, k, dim, largest, list(xv.shape),
                                         xv.dtype, xv.device)
        for u in users:
            u.replace_all_uses_with(values if u.args[1] == 0 else indices)
        changed += 1
    if changed:
        _recompile(g)


_MUL = (aten.mul.Tensor, aten.mul.Scalar)
_SHAPE = (aten.unsqueeze.default,)
_HANDLED = (aten.view_as_complex.default,) + _MUL + _SHAPE


def _val(node):
    """The fake value of a node, or the argument itself when it is not a node."""
    return node.meta.get("val", None) if isinstance(node, torch.fx.Node) else node


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


class ComplexToRealPairs(CustomGraphPass):
    """Complex values become their real [..., 2] pairs, whole component or nothing.

    Handles view_as_complex, mul and unsqueeze; any other complex op leaves the graph as is.
    """

    def uuid(self):
        return "pytorchsim-complex-to-real-pairs-2"

    def __call__(self, graph: torch.fx.Graph) -> None:
        complex_nodes = [n for n in graph.nodes if _is_complex(n)]
        if not complex_nodes:
            return

        for n in complex_nodes:
            if n.op != "call_function":
                return
            if n.target not in _HANDLED:
                return
            for user in n.users:
                if user.op == "output":
                    return
                if user.target is aten.view_as_real.default:
                    continue
                if user.target in _HANDLED:
                    continue
                return

        fake_mode = _fake_mode(graph)

        def _unwrap(a):
            if isinstance(a, (list, tuple)):
                return type(a)(_unwrap(x) for x in a)
            return _val(a)

        def call(target, *args):
            node = graph.call_function(target, args)
            if fake_mode is not None:
                with fake_mode:
                    node.meta["val"] = target(*[_unwrap(a) for a in args])
            return node

        pair = {}
        for n in complex_nodes:
            with graph.inserting_before(n):
                if n.target is aten.view_as_complex.default:
                    pair[n] = n.args[0]
                    continue

                if n.target in _SHAPE:
                    dim = n.args[1]
                    pair[n] = call(n.target, pair[n.args[0]],
                                   dim if dim >= 0 else dim - 1)
                    continue

                a, b = n.args[0], n.args[1]
                if _is_complex(a) and _is_complex(b):
                    pa, pb = pair[a], pair[b]
                    ar = call(aten.select.int, pa, -1, 0)
                    ai = call(aten.select.int, pa, -1, 1)
                    br = call(aten.select.int, pb, -1, 0)
                    bi = call(aten.select.int, pb, -1, 1)
                    re = call(aten.sub.Tensor, call(aten.mul.Tensor, ar, br),
                              call(aten.mul.Tensor, ai, bi))
                    im = call(aten.add.Tensor, call(aten.mul.Tensor, ar, bi),
                              call(aten.mul.Tensor, ai, br))
                    pair[n] = call(aten.cat.default,
                                   [call(aten.unsqueeze.default, re, -1),
                                    call(aten.unsqueeze.default, im, -1)], -1)
                else:
                    cx, other = (a, b) if _is_complex(a) else (b, a)
                    if isinstance(other, torch.fx.Node):
                        other = call(aten.unsqueeze.default, other, -1)
                    pair[n] = call(aten.mul.Tensor, pair[cx], other)

        for n in complex_nodes:
            for user in list(n.users):
                if user.target is aten.view_as_real.default:
                    user.replace_all_uses_with(pair[n])

        graph.eliminate_dead_code()


_COMPLEX_TO_REAL = ComplexToRealPairs()


def install():
    """Chain the post-grad passes onto Inductor's hook, each at most once."""
    install_once("pytorchsim-grouped-mm", npu_only(rewrite_grouped_mm))
    install_once("pytorchsim-topk", npu_only(rewrite_topk))
    install_once(_COMPLEX_TO_REAL.uuid(), npu_only(_COMPLEX_TO_REAL))
