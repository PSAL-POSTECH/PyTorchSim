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
    """Inspect a group of scheduler nodes and return {axis_index: boundaries}.

    `boundaries` is an ascending divisibility chain [1, b1, ..., E] of cut points
    for that axis: splitting the axis at these boundaries (mixed radix,
    `v = sum_i d_i * b_i`) makes every FloorDiv/ModularIndexing on it collapse to
    an affine combination of the split sub-vars. The cut points are gathered from
    the terms on the axis:
      - FloorDiv(v, k)            -> boundary k
      - ModularIndexing(v, k, m)  -> boundaries k and k*m   (the digit lives in [k, k*m))
    Only aligned terms count (the boundary must divide the extent E). If the
    collected boundaries for an axis do NOT form a divisibility chain (e.g.
    floor-by-2 and mod-by-3 on extent 6), the radices are incompatible -> the axis
    is left unsplit (its floor/mod stays for the misaligned/recompile path).

    axis_index is positional in the group's iteration space, so the same plan
    applies to every fused node sharing that space.
    """
    import collections
    bset = collections.defaultdict(set)     # axis -> set of boundary cut points
    ext_of = {}                             # axis -> extent
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
                    E = _as_int(body.var_ranges.get(base))
                    if E and E % k == 0:
                        bset[var_to_axis[base]].add(k); ext_of[var_to_axis[base]] = E
            for mi in expr.atoms(ModularIndexing):
                base, div, mod = mi.args
                k, m = _as_int(div), _as_int(mod)
                if base in var_to_axis and k and m:
                    E = _as_int(body.var_ranges.get(base))
                    if E and E % (k * m) == 0:
                        ax = var_to_axis[base]
                        if k > 1:
                            bset[ax].add(k)
                        if k * m < E:
                            bset[ax].add(k * m)
                        ext_of[ax] = E

    plan = {}
    for ax, bs in bset.items():
        E = ext_of[ax]
        chain = [1] + sorted(b for b in bs if 1 < b < E) + [E]
        # require a strict divisibility chain (each boundary divides the next).
        if len(chain) > 2 and all(chain[i + 1] % chain[i] == 0 for i in range(len(chain) - 1)):
            plan[ax] = chain

    # Validation aid: force-split the first even index axis even without floor/mod.
    # A floor-free index split is an identity transformation, so allclose must hold;
    # used to exercise the reduction pass-through path (no natural op produces a
    # floor on a reduction kernel's index axis). Off unless TORCHSIM_AXIS_SPLIT_FORCE.
    import os as _os
    if _os.environ.get("TORCHSIM_AXIS_SPLIT_FORCE"):
        for n in nodes:
            body = getattr(n, "_body", None)
            if body is None or not body.reduce_vars:
                continue
            for ax, v in enumerate(body.iter_vars):
                E = _as_int(body.var_ranges.get(v))
                if ax not in plan and E and E % 2 == 0 and E > 2:
                    plan[ax] = [1, 2, E]
                    break
    return plan


def build_split_body(node, plan, prefix="z"):
    """Rebuild node._body / sizes for the given split plan.

    Returns (body, (index_size, reduce_size)). Reindexes the EXISTING (already
    collapsed/reordered) node._body via LoopBody's copy path instead of re-tracing
    from the raw store function: pass the body as `fn` so LoopBody.__init__ takes
    _init_with_copy, which substitutes each original iter var with our expression
    and runs simplify_with_ranges. For a split axis the substitution
    v -> sum_i d_i * b_i (mixed radix over the boundary chain) makes every
    FloorDiv/ModularIndexing on it collapse to an affine combination of the d_i,
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
            bounds = plan[ax]                 # ascending chain [1, b1, ..., E]
            # one sub-var per segment: d_i has extent b_{i+1}/b_i, significance b_i.
            subs = []                         # (symbol, extent, significance) low->high
            expr = sympy.Integer(0)
            for i in range(len(bounds) - 1):
                seg_ext = bounds[i + 1] // bounds[i]
                nv = sympy_index_symbol(f"{prefix}{ctr}"); ctr += 1
                subs.append((nv, seg_ext, bounds[i]))
                expr = expr + nv * bounds[i]
            # iteration nest: most-significant (outermost) dim first.
            for nv, seg_ext, _sig in reversed(subs):
                iter_vars.append(nv)
                var_ranges[nv] = sympy.Integer(seg_ext)
                index_size.append(sympy.Integer(seg_ext))
            index_args.append(expr)
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
