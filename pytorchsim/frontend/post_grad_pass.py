"""The npu's post-grad pass: rewrites that keep ops with no npu kernel on the device.

pytorchsim_post_grad_pass is the device_custom_pass Inductor runs on npu graphs only, after its post-grad
passes: histc, polar, a row times a matrix, complex values as real pairs. topk and grouped_mm are
frontend/topk.py and frontend/grouped_mm.py.
"""

import torch
from torch import fx
from torch.fx import traceback as fx_traceback
from torch._inductor.custom_graph_pass import CustomGraphModulePass, get_hash_for_files

from pytorchsim import config

aten = torch.ops.aten
prims = torch.ops.prims

logger = config.setup_logger()


class _pytorchsim_post_grad_pass(CustomGraphModulePass):
    """The four rewrites in this order on an npu graph (polar before the complex values it makes); its uuid
    hashes this file, so an edit here is a new cache key."""

    def __call__(self, gm: fx.GraphModule) -> None:
        graph, fake_mode = gm.graph, _fake_mode(gm.graph)
        changed = (rewrite_aten_histc(graph, fake_mode)
                   + rewrite_aten_polar(graph, fake_mode)
                   + rewrite_aten_mm_addmm_row(graph, fake_mode)
                   + rewrite_aten_view_as_complex(graph, fake_mode))
        if changed:
            graph.eliminate_dead_code()
            gm.recompile()

    def uuid(self):
        return get_hash_for_files((__file__,))


pytorchsim_post_grad_pass = _pytorchsim_post_grad_pass()


def rewrite_aten_histc(graph, fake_mode) -> int:
    """Every aten.histc as a count on the device (_insert_histc); one without a fake value is left to aten,
    logged. Returns how many were rewritten."""
    def build(call, node):
        x = node.args[0]
        if _val(x) is None:
            logger.info("[post_grad_pass] histc without a fake value: left to aten")
            return None
        said = dict(zip(("bins", "min", "max"), node.args[1:]), **node.kwargs)
        return _insert_histc(call, x, _val(x), int(said.get("bins", 100)), said.get("min", 0), said.get("max", 0))
    return _replace(graph, fake_mode, graph.find_nodes(op="call_function", target=aten.histc.default), build)


def _insert_histc(call, x, xv, bins, lo, hi):
    """aten's histc: bin (x - lo) * bins / (hi - lo), integral in int64 and floating in x's dtype, the top
    edge in the last bin, outside [lo, hi] dropped; lo == hi takes x's range, widened by 1 if it is one value."""
    integral = not xv.is_floating_point()
    flat = call(prims.convert_element_type.default, call(aten.reshape.default, x, [-1]),
                torch.int64 if integral else xv.dtype)
    if lo == hi and xv.numel():
        lo, hi = call(aten.amin.default, flat, [0]), call(aten.amax.default, flat, [0])
        one = call(aten.eq.Tensor, lo, hi)
        lo, hi = (call(aten.where.self, one, call(aten.sub.Tensor, lo, 1), lo),
                  call(aten.where.self, one, call(aten.add.Tensor, hi, 1), hi))
        width = call(aten.sub.Tensor, hi, lo)
    else:
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        if integral:
            lo, hi = int(lo), int(hi)
        if hi <= lo:
            raise RuntimeError(f"torch.histc: max must be larger than min (min {lo}, max {hi})")
        width = hi - lo
    scaled = call(aten.mul.Tensor, call(aten.sub.Tensor, flat, lo), bins)
    if integral:
        pos = call(aten.div.Tensor_mode, scaled, width, rounding_mode="floor")
    else:
        pos = call(prims.convert_element_type.default, call(aten.div.Tensor, scaled, width), torch.int64)
    pos = call(aten.clamp_max.default, pos, bins - 1)
    ge, le = ((aten.ge.Tensor, aten.le.Tensor) if isinstance(lo, fx.Node)
              else (aten.ge.Scalar, aten.le.Scalar))
    inside = call(aten.logical_and.default, call(ge, flat, lo), call(le, flat, hi))
    seats = call(prims.iota.default, bins, start=0, step=1, dtype=torch.int64, device=xv.device,
                 requires_grad=False)
    hits = call(aten.logical_and.default,
                call(aten.eq.Tensor, call(aten.reshape.default, pos, [-1, 1]),
                     call(aten.reshape.default, seats, [1, -1])),
                call(aten.reshape.default, inside, [-1, 1]))
    return call(prims.convert_element_type.default, call(aten.sum.dim_IntList, hits, [0]), xv.dtype)


def rewrite_aten_polar(graph, fake_mode) -> int:
    """polar(abs, angle) as view_as_complex of the pair (abs*cos, abs*sin), built contiguous since
    view_as_complex needs a unit last stride; rewrite_aten_view_as_complex then takes it to real pairs."""
    def build(call, node):
        r, angle = node.args[:2]
        return call(aten.view_as_complex.default,
                    _join(call, call(aten.mul.Tensor, r, call(aten.cos.default, angle)),
                          call(aten.mul.Tensor, r, call(aten.sin.default, angle))))
    return _replace(graph, fake_mode, graph.find_nodes(op="call_function", target=aten.polar.default), build)


def rewrite_aten_mm_addmm_row(graph, fake_mode) -> int:
    """Replace a 1 x K row times a K x N matrix with the row broadcast down the matrix, a
    multiply and a sum over K: vector work, where the mm template would run it on the array."""
    def build(call, node):
        found = _row_times_matrix(node)
        if found is None:
            return None
        a, b, bias = found
        column = call(aten.view.default, a, [int(_val(a).shape[1]), 1])
        out = call(aten.sum.dim_IntList, call(aten.mul.Tensor, column, b), [0], True)
        return out if bias is None else call(aten.add.Tensor, out, bias)
    return _replace(graph, fake_mode, graph.find_nodes(op="call_function", target=aten.mm.default)
                    + graph.find_nodes(op="call_function", target=aten.addmm.default), build)


def _row_times_matrix(node):
    """(a, b, bias) of an mm or a plain addmm of real floating operands whose left one is a row, else None."""
    if node.target is aten.mm.default:
        (a, b), bias = node.args, None
    elif not node.kwargs:
        bias, a, b = node.args
    else:
        return None
    av, bv = _val(a), _val(b)
    if av is None or bv is None or av.dim() != 2 or bv.dim() != 2 or av.shape[0] != 1:
        return None
    if not (av.is_floating_point() and bv.is_floating_point()):
        return None
    return a, b, bias


_MUL = (aten.mul.Tensor, aten.mul.Scalar)
_SHAPE = (aten.unsqueeze.default,)
_HANDLED = (aten.view_as_complex.default,) + _MUL + _SHAPE


def rewrite_aten_view_as_complex(graph, fake_mode) -> int:
    """Each component of complex values (view_as_complex, mul, unsqueeze) becomes real [..., 2] pairs; a
    component with any other complex op stays as it is, logged with why. Returns the components rewritten."""
    changed = 0
    for component in _complex_components([n for n in graph.nodes if _is_complex(n)]):
        why = _why_not_real(component)
        if why is not None:
            logger.info("[post_grad_pass] %d complex nodes left to aten: %s", len(component), why)
            continue
        _to_real_pairs(graph, fake_mode, component)
        changed += 1
    return changed


def _complex_components(nodes):
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


def _why_not_real(component):
    """Why the component has no real-pair form, or None: a node outside _HANDLED (a placeholder or attribute
    among them), or a complex value read by anything but _HANDLED and view_as_real (the graph's output too)."""
    for n in component:
        if n.target not in _HANDLED:
            return f"{n.op} {n.target}"
        for user in n.users:
            if user.target is not aten.view_as_real.default and user.target not in _HANDLED:
                return f"{user.op} {user.name} reads {n.name}'s complex value"
    return None


def _to_real_pairs(graph, fake_mode, component):
    """Rewrite one eligible component: each complex value becomes its [..., 2] pair and every view_as_real of
    it reads the pair."""
    pair = {}
    for n in component:
        call = _inserter(graph, fake_mode, n)
        if n.target is aten.view_as_complex.default:
            pair[n] = n.args[0]
            continue
        if n.target in _SHAPE:
            dim = n.args[1]
            pair[n] = call(n.target, pair[n.args[0]], dim if dim >= 0 else dim - 1)
            continue
        a, b = n.args[0], n.args[1]
        cx, other = (a, b) if _is_complex(a) else (b, a)
        if _is_complex(other) or isinstance(other, complex):
            ar, ai = call(aten.select.int, pair[cx], -1, 0), call(aten.select.int, pair[cx], -1, 1)
            if isinstance(other, complex):
                br, bi = other.real, other.imag
            else:
                br, bi = call(aten.select.int, pair[other], -1, 0), call(aten.select.int, pair[other], -1, 1)
            re = call(aten.sub.Tensor, call(aten.mul.Tensor, ar, br), call(aten.mul.Tensor, ai, bi))
            im = call(aten.add.Tensor, call(aten.mul.Tensor, ar, bi), call(aten.mul.Tensor, ai, br))
            pair[n] = _join(call, re, im)
        else:
            if isinstance(other, fx.Node):
                other = call(aten.unsqueeze.default, other, -1)
            pair[n] = call(aten.mul.Tensor, pair[cx], other)

    for n in component:
        for user in list(n.users):
            if user.target is aten.view_as_real.default:
                user.replace_all_uses_with(pair[n])


def _replace(graph, fake_mode, nodes, build):
    """Each node of nodes for which build(call, node) returns a value is replaced by it and erased, call
    being _inserter's for that node. Returns how many were replaced."""
    changed = 0
    for node in nodes:
        out = build(_inserter(graph, fake_mode, node), node)
        if out is not None:
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            changed += 1
    return changed


def _inserter(graph, fake_mode, origin):
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


def _join(call, re, im):
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
