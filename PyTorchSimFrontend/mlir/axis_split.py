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


def _as_digit(expr):
    """If ``expr`` is a single-variable *digit extractor* -- any nesting of FloorDiv /
    ModularIndexing whose innermost argument is one symbol ``v`` -- return ``(v, div, mod)``
    meaning ``(v // div) % mod`` (``mod is None`` -> a pure ``v // div``). Otherwise None.

    Every such nesting collapses to a single (div, mod) by composing divisors, from four
    algebraic identities (v >= 0, all constants positive integers):
        FloorDiv(v//a,        b)      = v // (a*b)
        FloorDiv((v//a)%m,    b)      = (v // (a*b)) % (m//b)      if b   | m
        ModularIndexing(v//a,    b,m2)= (v // (a*b)) % m2
        ModularIndexing((v//a)%m, b,m2)= (v // (a*b)) % m2         if b*m2| m
    The divisibility guards make every rewrite provably equality-preserving. A multi-
    variable inner argument (e.g. torch.roll's v+shift) is not a digit extractor -> None.
    """
    if isinstance(expr, sympy.Symbol):
        return (expr, 1, None)
    if isinstance(expr, FloorDiv):
        inner, b = expr.args
        b, e = _as_int(b), _as_digit(expr.args[0])
        if e is None or b is None:
            return None
        v, a, m = e
        if m is None:
            return (v, a * b, None)
        if m % b == 0:
            return (v, a * b, m // b)
        return None
    if isinstance(expr, ModularIndexing):
        inner, b, m2 = expr.args
        b, m2, e = _as_int(b), _as_int(m2), _as_digit(expr.args[0])
        if e is None or b is None or m2 is None:
            return None
        v, a, m = e
        if m is None:
            return (v, a * b, m2)
        if m % (b * m2) == 0:
            return (v, a * b, m2)
        return None
    return None


def _rebuild_digit(v, a, m):
    """Canonical single-level form of ``(v // a) % m``."""
    x = v if a == 1 else FloorDiv(v, a)
    return x if m is None else ModularIndexing(v, a, m)


def flatten_nested_floormod(expr):
    """Collapse nested single-variable FloorDiv/ModularIndexing to one level.

    A composition of aligned reshapes on one iteration variable leaves a nested index like
    ModularIndexing(ModularIndexing(p, 1, 64), 1, 8) that neither sympy nor
    simplify_with_ranges reduces, so collect_boundaries skips its cut points (the inner base
    is not a bare var) and the affine-only DMA check later rejects it. Rewriting each nested
    digit extractor to its single-level (v // A) % M form (via _as_digit) exposes those cut
    points to axis-split. General and pattern-free -- no per-shape special cases.
    """
    try:
        atoms = expr.atoms(FloorDiv, ModularIndexing)
    except AttributeError:
        return expr
    replace = {}
    for atom in atoms:
        e = _as_digit(atom)
        if e is not None:
            canon = _rebuild_digit(*e)
            if canon != atom:
                replace[atom] = canon
    return expr.xreplace(replace) if replace else expr


# --- symbolic-aware boundary arithmetic ------------------------------------
# These reduce EXACTLY to the integer case when their operands are concrete, so
# static axis splitting is unchanged; they additionally accept symbolic size
# expressions (e.g. a flattened reshape extent E = M*N with divisor N), where a
# boundary that is a genuine product of dims divides the extent by construction.
# A dynamic dim symbol is created integer/positive, so sympy proves the
# divisibility (Mod(M*N, N) -> 0) and the quotient (cancel(M*N/N) -> M).

def _divides(d, E):
    """True iff d divides E. For concrete ints this is `E % d == 0`."""
    di, Ei = _as_int(d), _as_int(E)
    if di is not None and Ei is not None:
        return di != 0 and Ei % di == 0
    try:
        return bool(sympy.simplify(sympy.Mod(E, d)) == 0)
    except Exception:
        return False


def _eq(a, b):
    """Provable equality of two size exprs (structural for ints)."""
    ai, bi = _as_int(a), _as_int(b)
    if ai is not None and bi is not None:
        return ai == bi
    try:
        return bool(sympy.simplify(a - b) == 0)
    except Exception:
        return a == b


def _gt1(x):
    """True iff x is a non-trivial boundary (> 1). A symbolic dim is assumed > 1."""
    xi = _as_int(x)
    if xi is not None:
        return xi > 1
    return not _eq(x, sympy.Integer(1))


def _proper(b, E):
    """True iff b is a proper interior divisor of E: 1 < b < E and b | E."""
    bi, Ei = _as_int(b), _as_int(E)
    if bi is not None and Ei is not None:
        return 1 < bi < Ei and Ei % bi == 0
    return _gt1(b) and not _eq(b, E) and _divides(b, E)


def _quotient(a, b):
    """a / b as an exact int (concrete) or simplified sympy expr (symbolic)."""
    ai, bi = _as_int(a), _as_int(b)
    if ai is not None and bi is not None:
        return ai // bi
    return sympy.cancel(a / b)


def _as_size(x):
    """Wrap a concrete int as sympy.Integer; pass a sympy expr through unchanged
    (preserving its integer/positive assumptions)."""
    xi = _as_int(x)
    return sympy.Integer(xi) if xi is not None else x


def _ordered_chain(boundaries, E):
    """Order the proper divisors of E into a divisibility chain [1, ..., E], else None.

    Generalises the old `_is_chain` + numeric `sorted`: orders by the divisibility
    partial order (b_i precedes b_j iff b_i | b_j) rather than by numeric value, so
    symbolic boundaries (suffix-products of dims, e.g. N | M*N) chain correctly. For
    concrete ints this yields exactly the old ascending divisibility chain. Returns
    None when the boundaries do not form a TOTAL divisibility chain (the
    incompatible-radix / misaligned case), so the axis is left unsplit.
    """
    bs = []
    for b in boundaries:
        if _proper(b, E) and not any(_eq(b, x) for x in bs):
            bs.append(b)
    ordered = []
    remaining = list(bs)
    while remaining:
        # the divisibility-minimum is the unique element that divides all others.
        mins = [b for b in remaining
                if all(_divides(b, o) for o in remaining if not _eq(b, o))]
        if len(mins) != 1:
            return None  # no unique minimum -> incomparable -> not a chain
        ordered.append(mins[0])
        remaining = [o for o in remaining if not _eq(o, mins[0])]
    chain = [sympy.Integer(1)] + ordered + [_as_size(E)]
    for i in range(len(chain) - 1):
        if not _divides(chain[i], chain[i + 1]):
            return None
    return chain


def collect_boundaries(exprs, var_to_axis, var_ranges):
    """{axis_index: set(boundary cut points)} for the given index expressions.

    A FloorDiv(v, k) contributes boundary k; ModularIndexing(v, k, m) contributes
    k and k*m. Only aligned terms count (boundary divides the var extent). Shared
    by find_split_plan (fused LoopBody) and graph_copy (operand loaders). Boundaries
    and extents may be symbolic (dynamic reshape); divisibility is checked via
    `_divides`, so a symbolic divisor that is a genuine factor of the extent counts.
    """
    import collections
    bset = collections.defaultdict(set)
    for expr in exprs:
        expr = flatten_nested_floormod(expr)   # nested digit extractors -> single level
        for fd in expr.atoms(FloorDiv):
            base, div = fd.args
            if base in var_to_axis and _gt1(div):
                E = var_ranges.get(base)
                if E is not None and _divides(div, E):
                    bset[var_to_axis[base]].add(div)
        for mi in expr.atoms(ModularIndexing):
            base, div, mod = mi.args
            if base in var_to_axis:
                E = var_ranges.get(base)
                km = div * mod
                if E is not None and _divides(km, E):
                    ax = var_to_axis[base]
                    if _gt1(div):
                        bset[ax].add(div)
                    if _proper(km, E):
                        bset[ax].add(km)
    return bset


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
    Boundaries/extents may be symbolic (see _ordered_chain).

    axis_index is positional in the group's iteration space, so the same plan
    applies to every fused node sharing that space.
    """
    import collections
    bset = collections.defaultdict(set)     # axis -> set of boundary cut points
    ext_of = {}                             # axis -> extent (int or symbolic)
    for n in nodes:
        body = getattr(n, "_body", None)
        if body is None:
            continue
        var_to_axis = {v: i for i, v in enumerate(body.iter_vars)}
        nb = collect_boundaries(body.indexing_exprs.values(), var_to_axis, body.var_ranges)
        for ax, bs in nb.items():
            bset[ax] |= bs
            ext_of[ax] = body.var_ranges[body.iter_vars[ax]]

    plan = {}
    for ax, bs in bset.items():
        E = ext_of.get(ax)
        if E is None:
            continue
        # require a real, divisibility-chain split (incompatible radices -> skip).
        chain = _ordered_chain(bs, E)
        if chain is not None and len(chain) > 2:
            plan[ax] = chain

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
                seg_ext = _quotient(bounds[i + 1], bounds[i])
                nv = sympy_index_symbol(f"{prefix}{ctr}"); ctr += 1
                subs.append((nv, seg_ext, bounds[i]))
                expr = expr + nv * bounds[i]
            # iteration nest: most-significant (outermost) dim first.
            for nv, seg_ext, _sig in reversed(subs):
                iter_vars.append(nv)
                var_ranges[nv] = _as_size(seg_ext)
                index_size.append(_as_size(seg_ext))
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
    expr = flatten_nested_floormod(expr)   # collapse any residual single-var nested digit
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
