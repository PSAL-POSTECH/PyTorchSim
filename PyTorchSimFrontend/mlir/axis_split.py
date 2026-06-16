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
from torch._inductor.utils import sympy_index_symbol
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


def build_split_body(node, plan, prefix="z"):
    """Rebuild node._body / sizes for the given split plan.

    Returns (body, (index_size, reduce_size)). Reindexes the EXISTING (already
    collapsed/reordered) node._body via LoopBody's copy path instead of re-tracing
    from the raw store function: pass the body as `fn` so LoopBody.__init__ takes
    _init_with_copy, which substitutes each original iter var with our expression
    and runs simplify_with_ranges. For a split axis the substitution v -> outer*k
    + inner makes FloorDiv(v, k) collapse to `outer` (and ModularIndexing reduce),
    and reindexing the collapsed body keeps already-merged dims merged (no rank
    blow-up). indexing_from_args requires exactly one replacement expr per original
    var (index dims then reduce dims), flattened to len(body.var_ranges).
    """
    body = node._body
    orig_index_vars = list(body.iter_vars)
    orig_reduce_vars = list(body.reduce_vars)

    iter_vars = []
    index_args = []             # one expr per ORIGINAL index dim (substituted in)
    var_ranges = {}
    index_size = []
    ctr = 0

    for ax, v in enumerate(orig_index_vars):
        ext = body.var_ranges[v]
        if ax in plan:
            k = plan[ax]
            ext_i = _as_int(ext)
            outer = sympy_index_symbol(f"{prefix}{ctr}"); ctr += 1
            inner = sympy_index_symbol(f"{prefix}{ctr}"); ctr += 1
            iter_vars += [outer, inner]
            var_ranges[outer] = sympy.Integer(ext_i // k)
            var_ranges[inner] = sympy.Integer(k)
            index_size += [sympy.Integer(ext_i // k), sympy.Integer(k)]
            index_args.append(outer * k + inner)
        else:
            nv = sympy_index_symbol(f"{prefix}{ctr}"); ctr += 1
            iter_vars.append(nv)
            var_ranges[nv] = ext
            index_size.append(ext)
            index_args.append(nv)

    # Reduction dims pass through unchanged (a fresh symbol with the same range),
    # using the "r" prefix and kept after the index dims so the reduction axis
    # stays innermost (var_ranges is ordered iter-then-reduce; sizes splits on
    # len(iter_vars)). We do not split reduction dims here.
    reduce_vars = []
    reduce_size = []
    reduce_args = []
    for rctr, v in enumerate(orig_reduce_vars):
        ext = body.var_ranges[v]
        nv = sympy_index_symbol(f"r{rctr}")
        reduce_vars.append(nv)
        var_ranges[nv] = ext
        reduce_size.append(ext)
        reduce_args.append(nv)

    args = [index_args, reduce_args] if orig_reduce_vars else [index_args]
    new_body = LoopBody(body, args, var_ranges, iter_vars, reduce_vars)
    return new_body, (index_size, reduce_size)
