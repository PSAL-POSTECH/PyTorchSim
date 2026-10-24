"""The complex family: complex values as real [..., 2] pairs, one rule per complex op in _REAL_PAIR. It
walks the whole graph after the patterns, as a value's pair depends on the ops around it."""

import torch
from torch import fx
from torch.fx import traceback as fx_traceback

from pytorchsim import config
from pytorchsim.frontend.inductor.post_grad_pass import count, fake_value, fallback_route

aten = torch.ops.aten

logger = config.setup_logger()


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
                return fake_value(a)

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
            count("complex_real_pairs", "fallback")
            logger.info("[post_grad_pass] %d complex nodes stay: %s; complex ops run as %s",
                        len(component), why, fallback_route(component[0].target))
            continue
        to_real_pairs(graph, fake_mode, component)
        count("complex_real_pairs", "rewritten")
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


def _pair(call, re, im):
    """The [..., 2] real pair of re and im."""
    return call(aten.cat.default, [call(aten.unsqueeze.default, re, -1),
                                   call(aten.unsqueeze.default, im, -1)], -1)


def _is_complex(node):
    """Whether the node holds a complex tensor."""
    v = fake_value(node)
    return isinstance(v, torch.Tensor) and v.is_complex()
