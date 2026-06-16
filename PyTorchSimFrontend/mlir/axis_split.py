"""Aligned axis splitting at the Inductor scheduling layer.

Goal: guarantee the MLIR codegen sees only per-axis affine index expressions
(no FloorDiv / ModularIndexing). When an index expr contains FloorDiv(v, k) or
ModularIndexing(v, k, m) where `v` is a single iteration variable of extent E
and the divisor (resp. k*m) divides E, the floor/mod is *aligned*: splitting the
loop axis v into (outer, inner) with v = outer*k + inner makes it collapse to a
plain affine term (outer), at zero data-movement cost.

This is the cheap upstream tool of the affine-only contract. The misaligned case
(cat / non-factor reshape, divisor does not divide the extent) is NOT handled
here -- that needs graph-level copy insertion.

The rebuild reuses Inductor's own LoopBody machinery, exactly like
MLIRScheduling.revert_group: feed a split var_ranges + iter_vars and re-trace the
node's store function so the index expressions are regenerated over the new
iteration domain.
"""
import sympy
from torch._inductor.ir import LoopBody
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def find_split_plan(nodes):
    """Inspect a group of scheduler nodes and return {axis_index: divisor}.

    axis_index is positional in the group's iteration space (iter vars), so the
    same plan applies to every fused node sharing that space. Only aligned,
    statically-divisible splits are returned; dynamic / non-dividing terms are
    left for the misaligned (copy) path.
    """
    plan = {}
    for n in nodes:
        body = getattr(n, "_body", None)
        if body is None:
            continue
        var_to_axis = {v: i for i, v in enumerate(body.iter_vars)}
        for expr in body.indexing_exprs.values():
            for fd in expr.atoms(FloorDiv):
                base, div = fd.args
                k = _as_int(div)
                if base in var_to_axis and k and k > 1:
                    ext = _as_int(body.var_ranges.get(base))
                    if ext and ext % k == 0:
                        plan.setdefault(var_to_axis[base], k)
            for mi in expr.atoms(ModularIndexing):
                base, div, mod = mi.args
                k, m = _as_int(div), _as_int(mod)
                if base in var_to_axis and k and m:
                    ext = _as_int(body.var_ranges.get(base))
                    if ext and ext % (k * m) == 0:
                        # split off the inner block of size k so FloorDiv(.,k)->outer
                        plan.setdefault(var_to_axis[base], k)
    return plan


def build_split_body(node, plan, prefix="s"):
    """Rebuild node._body / sizes for the given split plan.

    Returns (body, (index_size, reduce_size)). Mirrors revert_group: re-trace the
    store function with index args where a split output dim `ax` is fed the
    expression outer*k + inner, and var_ranges carries the two new vars.
    """
    inode = node.node
    size = inode.data.get_size()
    reduction_size = inode.data.get_reduction_size()

    iter_vars = []
    fn_index_args = []          # one expr per ORIGINAL output dim
    var_ranges = {}
    index_size = []
    ctr = 0

    for ax, ext in enumerate(size):
        if ax in plan:
            k = plan[ax]
            ext_i = _as_int(ext)
            outer = sympy.Symbol(f"{prefix}{ctr}"); ctr += 1
            inner = sympy.Symbol(f"{prefix}{ctr}"); ctr += 1
            iter_vars += [outer, inner]
            var_ranges[outer] = sympy.Integer(ext_i // k)
            var_ranges[inner] = sympy.Integer(k)
            index_size += [sympy.Integer(ext_i // k), sympy.Integer(k)]
            fn_index_args.append(outer * k + inner)
        else:
            v = sympy.Symbol(f"{prefix}{ctr}"); ctr += 1
            iter_vars.append(v)
            var_ranges[v] = ext
            index_size.append(ext)
            fn_index_args.append(v)

    reduce_vars = []
    reduce_size = []
    for ext in reduction_size:
        v = sympy.Symbol(f"{prefix}{ctr}"); ctr += 1
        reduce_vars.append(v)
        var_ranges[v] = ext
        reduce_size.append(ext)

    store_fn = inode.get_store_function()
    fn_args = [fn_index_args, reduce_vars] if inode.get_reduction_type() else [fn_index_args]
    body = LoopBody(store_fn, fn_args, var_ranges, iter_vars, reduce_vars)
    return body, (index_size, reduce_size)
