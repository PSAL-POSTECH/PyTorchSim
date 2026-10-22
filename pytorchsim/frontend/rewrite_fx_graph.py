"""Our rewrites of the FX graph, so ops with no npu kernel stay on the device.

npu_post_grad_pass is the device_custom_pass Inductor runs on npu graphs only, after its post-grad
passes: histc, polar, a row times a matrix, complex values as real pairs. topk and grouped_mm are
frontend/topk.py and frontend/grouped_mm.py.
"""

import torch
from torch import fx
from torch._inductor.custom_graph_pass import CustomGraphModulePass, get_hash_for_files

from pytorchsim import config

aten = torch.ops.aten
prims = torch.ops.prims

logger = config.setup_logger()


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


def _emitter(graph):
    """call(target, *args, **kwargs): a node at the graph's insert point with its fake value set, since
    nothing after the device pass sets it."""
    fake_mode = _fake_mode(graph)

    def unwrap(a):
        if isinstance(a, (list, tuple)):
            return type(a)(unwrap(x) for x in a)
        if isinstance(a, dict):
            return {k: unwrap(v) for k, v in a.items()}
        return _val(a)

    def call(target, *args, **kwargs):
        node = graph.call_function(target, args, kwargs)
        if fake_mode is not None:
            with fake_mode:
                node.meta["val"] = target(*unwrap(args), **unwrap(kwargs))
        return node

    return call


def _join(call, re, im):
    """The [..., 2] real pair of re and im."""
    return call(aten.cat.default, [call(aten.unsqueeze.default, re, -1),
                                   call(aten.unsqueeze.default, im, -1)], -1)


def rewrite_aten_histc(graph) -> None:
    """histc of an integral tensor whose bins are its values (bins N, min 0, max N-1) as one compare and one
    count in the input's dtype; other integral edges count at float32 and cast back, an extern call, logged."""
    call = _emitter(graph)
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.histc.default:
            continue
        x = node.args[0]
        xv = _val(x)
        said = dict(zip(("bins", "min", "max"), node.args[1:]), **node.kwargs)
        bins, lo, hi = int(said.get("bins", 100)), said.get("min", 0), said.get("max", 0)
        if xv is None or xv.is_floating_point():
            logger.info("[rewrite_fx_graph] histc %s: left to aten (no npu kernel)",
                        None if xv is None else xv.dtype)
            continue
        with graph.inserting_before(node):
            if lo == 0 and hi == bins - 1:
                seats = call(prims.iota.default, bins, start=0, step=1, dtype=xv.dtype,
                             device=xv.device, requires_grad=False)
                hits = call(aten.eq.Tensor, call(aten.reshape.default, x, [-1, 1]),
                            call(aten.reshape.default, seats, [1, -1]))
                out = call(prims.convert_element_type.default,
                           call(aten.sum.dim_IntList, hits, [0]), xv.dtype)
            else:
                logger.info("[rewrite_fx_graph] histc %s bins %d [%s, %s]: an extern call at float32",
                            xv.dtype, bins, lo, hi)
                xf = call(prims.convert_element_type.default, x, torch.float32)
                out = call(prims.convert_element_type.default,
                           call(aten.histc.default, xf, bins, lo, hi), xv.dtype)
        node.replace_all_uses_with(out)
        graph.erase_node(node)


def rewrite_aten_polar(graph) -> None:
    """polar(abs, angle) as view_as_complex of the pair (abs*cos, abs*sin), built contiguous since
    view_as_complex needs a unit last stride; rewrite_aten_view_as_complex then takes it to real pairs."""
    call = _emitter(graph)
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.polar.default:
            continue
        r, angle = node.args[0], node.args[1]
        with graph.inserting_before(node):
            pair = _join(call, call(aten.mul.Tensor, r, call(aten.cos.default, angle)),
                         call(aten.mul.Tensor, r, call(aten.sin.default, angle)))
            out = call(aten.view_as_complex.default, pair)
        node.replace_all_uses_with(out)
        graph.erase_node(node)


def _row_times_matrix(node):
    """(a, b, bias) of an mm or a plain addmm whose left operand is one row, else None."""
    if node.op != "call_function":
        return None
    if node.target is aten.mm.default:
        (a, b), bias = node.args, None
    elif node.target is aten.addmm.default and not node.kwargs:
        bias, a, b = node.args
    else:
        return None
    av, bv = _val(a), _val(b)
    if av is None or bv is None or av.dim() != 2 or bv.dim() != 2 or av.shape[0] != 1:
        return None
    return a, b, bias


def rewrite_aten_mm_addmm_row(graph) -> None:
    """Replace a 1 x K row times a K x N matrix with the row broadcast down the matrix, a
    multiply and a sum over K: vector work, where the mm template would run it on the array."""
    call = _emitter(graph)
    for node in list(graph.nodes):
        found = _row_times_matrix(node)
        if found is None:
            continue
        a, b, bias = found
        k = int(_val(a).shape[1])
        with graph.inserting_before(node):
            column = call(aten.view.default, a, [k, 1])
            out = call(aten.sum.dim_IntList, call(aten.mul.Tensor, column, b), [0], True)
            if bias is not None:
                out = call(aten.add.Tensor, out, bias)
        node.replace_all_uses_with(out)
        graph.erase_node(node)


_MUL = (aten.mul.Tensor, aten.mul.Scalar)
_SHAPE = (aten.unsqueeze.default,)
_HANDLED = (aten.view_as_complex.default,) + _MUL + _SHAPE


def rewrite_aten_view_as_complex(graph) -> None:
    """Complex values become their real [..., 2] pairs, whole component or nothing.

    Handles view_as_complex, mul and unsqueeze; any other complex op leaves the graph as is.
    """
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

    call = _emitter(graph)
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
            cx, other = (a, b) if _is_complex(a) else (b, a)
            if _is_complex(other) or isinstance(other, complex):
                pa = pair[cx]
                ar = call(aten.select.int, pa, -1, 0)
                ai = call(aten.select.int, pa, -1, 1)
                if isinstance(other, complex):
                    br, bi = other.real, other.imag
                else:
                    br = call(aten.select.int, pair[other], -1, 0)
                    bi = call(aten.select.int, pair[other], -1, 1)
                re = call(aten.sub.Tensor, call(aten.mul.Tensor, ar, br),
                          call(aten.mul.Tensor, ai, bi))
                im = call(aten.add.Tensor, call(aten.mul.Tensor, ar, bi),
                          call(aten.mul.Tensor, ai, br))
                pair[n] = _join(call, re, im)
            else:
                if isinstance(other, torch.fx.Node):
                    other = call(aten.unsqueeze.default, other, -1)
                pair[n] = call(aten.mul.Tensor, pair[cx], other)

    for n in complex_nodes:
        for user in list(n.users):
            if user.target is aten.view_as_real.default:
                user.replace_all_uses_with(pair[n])

    graph.eliminate_dead_code()


REWRITES = (rewrite_aten_histc, rewrite_aten_polar, rewrite_aten_mm_addmm_row, rewrite_aten_view_as_complex)


class _npu_post_grad_pass(CustomGraphModulePass):
    """REWRITES in order on an npu graph; its uuid hashes this file, so an edit here is a new cache key."""

    def __call__(self, gm: fx.GraphModule) -> None:
        for rewrite in REWRITES:
            rewrite(gm.graph)
        gm.graph.eliminate_dead_code()
        gm.recompile()

    def uuid(self):
        return get_hash_for_files((__file__,))


npu_post_grad_pass = _npu_post_grad_pass()
