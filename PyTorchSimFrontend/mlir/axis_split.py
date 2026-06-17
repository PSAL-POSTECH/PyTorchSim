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


def collect_boundaries(exprs, var_to_axis, var_ranges):
    """{axis_index: set(boundary cut points)} for the given index expressions.

    A FloorDiv(v, k) contributes boundary k; ModularIndexing(v, k, m) contributes
    k and k*m. Only aligned terms count (boundary divides the var extent). Shared
    by find_split_plan (fused LoopBody) and graph_copy (operand loaders).
    """
    import collections
    bset = collections.defaultdict(set)
    for expr in exprs:
        for fd in expr.atoms(FloorDiv):
            base, div = fd.args
            k = _as_int(div)
            if base in var_to_axis and k and k > 1:
                E = _as_int(var_ranges.get(base))
                if E and E % k == 0:
                    bset[var_to_axis[base]].add(k)
        for mi in expr.atoms(ModularIndexing):
            base, div, mod = mi.args
            k, m = _as_int(div), _as_int(mod)
            if base in var_to_axis and k and m:
                E = _as_int(var_ranges.get(base))
                if E and E % (k * m) == 0:
                    ax = var_to_axis[base]
                    if k > 1:
                        bset[ax].add(k)
                    if k * m < E:
                        bset[ax].add(k * m)
    return bset


def _is_chain(boundaries, E):
    """True iff [1, sorted(boundaries in (1,E)), E] is a divisibility chain."""
    chain = [1] + sorted(b for b in boundaries if 1 < b < E) + [E]
    return all(chain[i + 1] % chain[i] == 0 for i in range(len(chain) - 1))


def ledger(nodes, plan):
    """Classify every FloorDiv/ModularIndexing in the kernel against `plan`.

    Returns a list of (op_name, reason, term_str) for the terms NOT covered by
    axis-split, so we can measure how often the graph-copy cases (incompatible
    radix / non-dividing / multi-axis / dynamic) actually reach codegen. Read-only.
    Reasons: covered terms are omitted; uncovered ones are
      multi_axis_arg     - floor/mod argument is not a single iter var (case 7)
      non_dividing       - divisor (or k*m) does not divide the extent (case 6)
      incompatible_radix - single var, divides, but boundaries did not form a
                           divisibility chain so the axis was left unsplit (case 5)
      dynamic            - symbolic divisor/extent
    """
    rows = []

    def classify(base, k, m, var_to_axis, var_ranges):
        if not (isinstance(base, sympy.Symbol) and base in var_to_axis):
            return None if False else "multi_axis_arg"
        ax = var_to_axis[base]
        E = _as_int(var_ranges.get(base))
        if k is None or E is None or (m is not None and _as_int(m) is None):
            return "dynamic"
        if ax in plan:
            return "covered"
        period = k if m is None else k * _as_int(m)
        if period and E % period != 0:
            return "non_dividing"
        return "incompatible_radix"

    for n in nodes:
        body = getattr(n, "_body", None)
        if body is None:
            continue
        op = n.get_name() if hasattr(n, "get_name") else "?"
        var_to_axis = {v: i for i, v in enumerate(body.iter_vars)}
        for expr in body.indexing_exprs.values():
            for fd in expr.atoms(FloorDiv):
                r = classify(fd.args[0], _as_int(fd.args[1]), None, var_to_axis, body.var_ranges)
                if r and r != "covered":
                    rows.append((op, r, str(fd)))
            for mi in expr.atoms(ModularIndexing):
                r = classify(mi.args[0], _as_int(mi.args[1]), mi.args[2], var_to_axis, body.var_ranges)
                if r and r != "covered":
                    rows.append((op, r, str(mi)))
    return rows


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
        nb = collect_boundaries(body.indexing_exprs.values(), var_to_axis, body.var_ranges)
        for ax, bs in nb.items():
            bset[ax] |= bs
            ext_of[ax] = _as_int(body.var_ranges[body.iter_vars[ax]])

    plan = {}
    for ax, bs in bset.items():
        E = ext_of[ax]
        # require a real, divisibility-chain split (incompatible radices -> skip).
        if E and any(1 < b < E for b in bs) and _is_chain(bs, E):
            plan[ax] = [1] + sorted(b for b in bs if 1 < b < E) + [E]

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

    # A split may push the per-axis index rank past 4. The resulting >4D logical tile
    # is peeled into <=4D physical descriptors by the decompose-transfer pass (an
    # affine.for nest carrying the lane-banked physical SRAM offset), so there is no
    # rank cap here.
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
    new_body.indexing_exprs = {
        name: _fold_with_ranges(e, var_ranges)
        for name, e in new_body.indexing_exprs.items()
    }
    return new_body, (index_size, reduce_size)


def _fold_with_ranges(expr, var_ranges):
    """Fold residual FloorDiv/ModularIndexing that simplify_with_ranges missed.

    A mixed-radix split leaves terms like FloorDiv(z1 + 4*z2, 12); these are 0 by
    construction (the lower digits sum below the boundary), but the Inductor
    simplifier cannot prove a multi-term numerator < divisor. We prove it directly
    from the split sub-var ranges via bound_sympy:
      FloorDiv(num, d)            -> 0          if 0 <= num < d
      ModularIndexing(num, k, m)  -> num // k   if 0 <= num < k*m   (mod is a no-op)
    Iterated to a fixpoint (folding a mod can expose a foldable floor).
    """
    from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
    ranges = {}
    for v, sz in var_ranges.items():
        e = _as_int(sz)
        if e is not None and e >= 1:
            ranges[v] = ValueRanges(0, e - 1)
    if not ranges:
        return expr

    def vr(num):
        try:
            return bound_sympy(num, ranges)
        except Exception:
            return None

    for _ in range(8):
        changed = False
        for fd in list(expr.atoms(FloorDiv)):
            num, div = fd.args
            d = _as_int(div)
            b = vr(num) if d else None
            if b is not None and b.lower >= 0 and b.upper < d:
                expr = expr.subs(fd, sympy.Integer(0)); changed = True
        for mi in list(expr.atoms(ModularIndexing)):
            num, k, m = mi.args
            ki, mi_ = _as_int(k), _as_int(m)
            b = vr(num) if (ki and mi_) else None
            if b is not None and b.lower >= 0 and b.upper < ki * mi_:
                expr = expr.subs(mi, FloorDiv(num, k)); changed = True
        if not changed:
            break
    return expr
