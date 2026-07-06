"""Graph-copy (relayout) for incompatible-radix operands.

When an elementwise consumer reads two operands whose floor/mod groupings on a
shared axis are incompatible (the boundary cut points do not form a divisibility
chain, e.g. floor-by-2 and mod-by-3 on extent 6), axis-split cannot linearize the
fused index. We `realize()` the cheaper operand at the consumer's lowering, which
materializes it as a contiguous buffer; the consumer then reads it affine and only
the other (single, compatible) grouping remains for axis-split to handle.

Detection reuses axis_split.collect_boundaries on each operand's loader index, so
it is the same precise radix analysis used at the scheduling layer -- not an FX
view-chain heuristic. The hook wraps the already-registered lowering entries (the
make_pointwise results), so it sees every elementwise consumer in one place. The
realize() (not a clone, which Inductor inlines) is what actually forces the buffer
boundary; see the PoC notes in docs.

Behavior-neutral unless a genuine incompatible-radix conflict is detected.
"""
from torch._inductor import lowering as L
from torch._inductor import dependencies
from torch._inductor import ir
from torch._inductor.ir import TensorBox
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from . import axis_split


def _has_multivar_floormod(exprs):
    """True if any FloorDiv/ModularIndexing argument spans >1 loop variable
    (case 7: cross-axis floor/mod that axis-split cannot split)."""
    for e in exprs:
        for f in list(e.atoms(FloorDiv)) + list(e.atoms(ModularIndexing)):
            if len(f.args[0].free_symbols) > 1:
                return True
    return False


def _numel(tb):
    n = 1
    for s in tb.get_size():
        v = axis_split._as_int(s)
        if v is None:
            return float("inf")
        n *= v
    return n


def _relayout_args(args):
    """Return a modified args list with one operand replaced by a forced copy when
    it needs relayout, or None to leave args unchanged. The copy uses
    ExternKernel.copy_input (a realized identity Pointwise) -- this materializes
    *views* too, unlike StorageBox.realize() which is a no-op on a ReinterpretView.
    The copy kernel iterates the operand's own (contiguous) shape, so its index
    collapses to single-var and axis-split handles it; the consumer then reads the
    copy affine."""
    pos = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    if not pos:
        return None
    tbs = [args[i] for i in pos]
    # Output/iteration shape = the broadcast of all operands (the largest rank,
    # max per dim). For a single-operand consumer (e.g. a reduction reading a
    # multi-var-view input) this is just that operand's shape -- still enough to
    # detect a multi-var floor and copy_input it (case 7); the 2-operand radix
    # conflict (case 5) naturally needs >=2 operands.
    # Per-dim max extent over the max-rank operands (order-independent). Picking a
    # single operand by rank alone (max key=len) would, for two equal-rank operands
    # with different per-dim extents, take a broadcast-from operand's smaller shape
    # and then miss the genuine conflict on the broadcast-to dim.
    maxrank = max(len(t.get_size()) for t in tbs)
    full = [t.get_size() for t in tbs if len(t.get_size()) == maxrank]
    ranges = [max((s[d] for s in full), key=lambda v: (axis_split._as_int(v) or -1))
              for d in range(maxrank)]
    extents = [axis_split._as_int(s) for s in ranges]
    if not extents or any(e is None for e in extents):
        return None                              # scalar / dynamic -> skip

    # Only true elementwise consumers: each operand is broadcast-compatible with the
    # output (same rank, every dim is 1 or == the output extent). This admits
    # broadcasting operands (e.g. y[8,1] into [8,3]) while excluding mm/bmm/cat-style
    # ops whose operands differ in a non-broadcast way.
    for tb in tbs:
        sz = [axis_split._as_int(s) for s in tb.get_size()]
        if len(sz) != len(extents) or any(
            d is not None and d != 1 and d != e for d, e in zip(sz, extents)
        ):
            return None

    # Trace each operand's loader to get its read indices (sympy) over the shared
    # output iteration; make_loader returns a value, so extract_read_writes is what
    # gives the index expressions. range_vars are positional per output axis, so the
    # axis numbering is consistent across operands.
    per_bnd = []                                 # [{axis: boundary set}] per operand
    per_mv = []                                  # [bool] operand has multi-var floor/mod
    for tb in tbs:
        try:
            rw = dependencies.extract_read_writes(tb.make_loader(), list(ranges))
        except Exception:
            per_bnd.append({})
            per_mv.append(False)
            continue
        v2a = {v: i for i, v in enumerate(rw.range_vars)}
        exprs = [r.index for r in rw.reads if hasattr(r, "index")]
        b = axis_split.collect_boundaries(exprs, v2a, rw.var_ranges)
        mv = _has_multivar_floormod(exprs)
        per_bnd.append(b)
        per_mv.append(mv)

    victim = None

    # Case 5 -- incompatible radices on a shared axis between two operands.
    for axis, E in enumerate(extents):
        contrib = [(i, per_bnd[i][axis]) for i in range(len(tbs)) if per_bnd[i].get(axis)]
        if len(contrib) < 2:
            continue                             # single grouping -> axis-split handles
        union = {b for _, s in contrib for b in s}
        if axis_split._is_chain(union, E):
            continue                             # compatible -> axis-split handles
        victim = min(contrib, key=lambda c: _numel(tbs[c[0]]))[0]
        break

    # Case 7 -- an operand whose floor/mod argument spans multiple consumer axes
    # (e.g. (3*p0+p1)//4 from a transpose+reshape feeding a broadcast/softmax that
    # keeps the dims separate). axis-split cannot split a multi-var argument.
    if victim is None:
        mv_ops = [i for i in range(len(tbs)) if per_mv[i]]
        if mv_ops:
            victim = min(mv_ops, key=lambda i: _numel(tbs[i]))

    if victim is None:
        return None
    new = list(args)
    p = pos[victim]
    new[p] = ir.ExternKernel.copy_input(args[p])
    return new


def install():
    """Wrap registered lowering entries to insert relayout. Idempotent. Call once
    at backend import (after torch._inductor.lowering is populated -- make_pointwise
    runs at import to build the entries, so we wrap the entries, not the factory)."""
    if getattr(L, "_torchsim_relayout_installed", False):
        return
    for key, fn in list(L.lowerings.items()):
        def wrap(orig):
            def wrapped(*a, **k):
                try:
                    na = _relayout_args(a)
                except Exception:
                    na = None                    # detection must never break lowering
                if na is not None:
                    a = na
                return orig(*a, **k)
            return wrapped
        L.lowerings[key] = wrap(fn)
    L._torchsim_relayout_installed = True
