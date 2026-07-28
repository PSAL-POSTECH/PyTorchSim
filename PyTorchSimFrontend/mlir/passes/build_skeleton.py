"""build_skeleton pass (C2): reduce a kernel's post-vcix MLIR to the
*skeleton + API* form, in place.

The trace pipeline (docs/design/togsim_cpp_trace.md) compiles a kernel to a
shape-parametric C++ trace producer. The producer is just the kernel's loop
skeleton with the data computation replaced by calls to the event-based runtime
API. This pass performs that reduction at the MLIR level:

  * `togsim.transfer`   -> `togsim.dma(...) {tag_id, is_async, ...}` carrying the
                            runtime tag index operand (`%tag[%idx]`).
  * `togsim.wait`   -> `togsim.memory_barrier(tag_idx) {tag_id, write_bufs}`,
                            the explicit async-DMA sync. It pairs with its dma by
                            the RUNTIME tag slot (tag_id + the tag index), not a
                            compile-time id: one static dma op runs once per loop
                            iteration with a different `%tag[%idx]`, so only the
                            runtime slot can pair iteration i's dma with its wait.
  * each compute node   -> a single `togsim.compute {tile_id, compute_type}`
  * everything else      -> removed by a use-based DCE, keeping the loops and the
                            index/address arithmetic the survivors depend on.

It reuses build_tog's traversal (`TogBuilder` / `_build`): loops, DMAs and
compute blocks are already identified there, each with a back-pointer to its
MLIR op(s), so this pass only adds the *rewrite*. Keeping a single traversal
guarantees the skeleton and the legacy TOG see the same structure.

Counterpart to `build_tog.build_tog_and_mutate`.

The DCE is safe by construction: it never erases an op whose results still have
uses, so at worst it leaves extra ops in the dump (visible for diagnosis) rather
than producing invalid IR.

Requires the MLIR Python bindings (importing `build_tog` pulls in `mlir.ir`).
"""

from . import togsim_ops as ts
from ._mlir_util import walk_ops, i32, i64, i64_array, str_attr
from .build_tog import (
    ir,
    TogBuilder,
    _build,
    _reset_ids,
    _find_kernel,
    _value_key,
    TOGDMANode,
    TOGDMAWaitNode,
    _COMPUTE_TYPE_NAME,
)

#: Marker op names for the passes/__init__ fast-path (skip parsing if absent).
MARKERS = ("togsim.transfer", "togsim.wait")

#: Ops the DCE must never remove (loops, terminators, our API ops).
_KEEP = {
    "affine.for", "scf.for", "scf.while",
    "affine.yield", "scf.yield", "func.return",
    ts.DMA, ts.COMPUTE, ts.MEMORY_BAR,
}


def _kernel_block(module):
    func_op = _find_kernel(module)
    if func_op is None:
        return None
    return func_op.regions[0].blocks[0]


# ---------------------------------------------------------------------------
# op construction
# ---------------------------------------------------------------------------
def _arg_id_of(base_addr):
    """Tensor func-arg ordinal from a build_tog base name ("arg3" -> 3); -1 if
    it is not a plain block-arg base."""
    s = str(base_addr)
    return int(s[3:]) if s.startswith("arg") and s[3:].isdigit() else -1


def _emit_dma(ctx, dma_node, tag_id, dram_index, tag_index, read_bufs, write_bufs):
    """Insert a `togsim.dma` before the original `togsim.transfer`.

    `tag_id` is the identity of this DMA's tag memref. An async DMA pairs with
    its `togsim.memory_barrier` (the original togsim.wait) by the RUNTIME tag slot
    -- (tag_id, tag_index) -- not a compile-time identifier: one static dma op runs
    once per loop iteration, each with a different runtime `%tag[%idx]` slot, so
    only a runtime key can pair iteration i's dma with iteration i's wait.

    `dram_index` is the original linear DRAM index Value (the `affine.apply`
    result that indexed the tensor in the `togsim.transfer`) -- carried as an
    operand so the DCE keeps the address arithmetic live and the C4 lowering can
    compute the real `base_addr = base[arg_id] + index*elem` (P3, approach A).

    `tag_index` is the original SRAM tag index Value (`%tag[%idx]`), carried as a
    second operand: the runtime tag slot, used both to pair with the barrier and
    for the double-buffer / SRAM-capacity (WAR) model.
    Operand order: [dram_index, tag_index] (each omitted if absent)."""
    op = dma_node.op
    attrs = {
        ts.ATTR_DIR: i32(ts.DIR_STORE if dma_node.is_write else ts.DIR_LOAD),
        ts.ATTR_DIMS: i64_array(dma_node.tile_size),
        ts.ATTR_STRIDES: i64_array(dma_node.tile_stride),
        ts.ATTR_ELEM_BITS: i32(dma_node.element_size),
        ts.ATTR_IS_ASYNC: ir.BoolAttr.get(bool(dma_node.is_async)),
        ts.ATTR_TAG_ID: i32(tag_id),
        ts.ATTR_ARG_ID: i32(_arg_id_of(dma_node.base_addr)),
        "base": str_attr(dma_node.base_addr),
        # SRAM spad this DMA touches (load writes it, store reads it) -- sec 10.
        ts.ATTR_READ_BUFS: i64_array(read_bufs),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    operands = [v for v in (dram_index, tag_index) if v is not None]
    ir.Operation.create(
        ts.DMA,
        results=[],
        operands=operands,
        attributes=attrs,
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(op),
    )


def _emit_memory_bar(ctx, anchor_op, tag_id, tag_index, write_bufs):
    """Insert a `togsim.memory_barrier` before `anchor_op` -- the explicit
    async-DMA sync that was the original `togsim.wait`. It pairs with its
    async `togsim.dma` by the RUNTIME tag slot (tag_id + tag_index), and carries
    the SRAM buffer that dma loaded so consumers gate on data-arrival, not on the
    async dma's issue-complete."""
    attrs = {
        ts.ATTR_TAG_ID: i32(tag_id),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    operands = [tag_index] if tag_index is not None else []
    ir.Operation.create(
        ts.MEMORY_BAR, results=[], operands=operands, attributes=attrs,
        loc=ir.Location.unknown(ctx), ip=ir.InsertionPoint(anchor_op))


def _flatten_add(expr):
    """Top-level additive summands of an AffineExpr (`.lhs`/`.rhs` come back typed
    as the base AffineExpr, so use the `isinstance`/cast pattern, not Python
    isinstance)."""
    if ir.AffineAddExpr.isinstance(expr):
        a = ir.AffineAddExpr(expr)
        return _flatten_add(a.lhs) + _flatten_add(a.rhs)
    return [expr]


def _neg_coeff_dim(summand):
    """If `summand` is `dim * c` with a negative constant `c`, return that dim's
    position; else None. lower_to_vcix tags each accumulation (reduction) loop var
    with coefficient -1 in the togsim.wait tag index -- a SENTINEL marking the
    reduction axis, not an arithmetic offset (legacy TileGraphParser skips stride
    -1 for the same reason)."""
    if not ir.AffineMulExpr.isinstance(summand):
        return None
    mul = ir.AffineMulExpr(summand)
    l, r = mul.lhs, mul.rhs
    dim = l if ir.AffineDimExpr.isinstance(l) else (r if ir.AffineDimExpr.isinstance(r) else None)
    con = l if ir.AffineConstantExpr.isinstance(l) else (r if ir.AffineConstantExpr.isinstance(r) else None)
    if dim is None or con is None or ir.AffineConstantExpr(con).value >= 0:
        return None
    return ir.AffineDimExpr(dim).position


def _strip_accum_terms(ctx, tag_index, anchor_op):
    """Return a tag-index Value with the accumulation-marked (-1 coefficient) terms
    dropped, so a memory_barrier waits on the SAME subtile slot its async load
    wrote.

    The wait tag index built by lower_to_vcix carries `-acc_iv` for each reduction
    loop var; the matching load index (dma_fine_grained) is subtile-only. Without
    this, at reduction iteration > 0 the producer EVALUATES `-acc_iv` to a negative
    slot, so the recorded barrier slot diverges from the load slot and the runtime
    tag pairing fails (TOGSim aborts with "Key does not exist in ... tag table").
    Dropping the -1 terms mirrors legacy TileGraphParser.cc, which skips stride -1
    and routes the reduction axis to a separate accum tag component; here the
    per-iteration tag alloc (dma_fine_grained) already separates the reductions, so
    the barrier only needs the subtile slot.

    Falls through (returns `tag_index` unchanged) for anything that is not an
    affine.apply whose single result carries such a term -- e.g. the single-tile
    case, whose index has no reduction term."""
    if tag_index is None:
        return None
    try:
        apply_op = tag_index.owner
        if apply_op.name != "affine.apply":
            return tag_index
        amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
    except Exception:
        return tag_index
    if amap.n_dims == 0 or amap.n_symbols != 0 or len(amap.results) != 1:
        return tag_index
    expr = amap.results[0]
    dropped = sorted({p for p in (_neg_coeff_dim(s) for s in _flatten_add(expr))
                      if p is not None})
    if not dropped:
        return tag_index
    n = amap.n_dims
    kept = [i for i in range(n) if i not in dropped]
    new_pos = {old: i for i, old in enumerate(kept)}
    # compose the original expr with a selector that sends each dropped dim to 0
    # and renumbers the kept dims 0..k-1.
    sel = [ir.AffineConstantExpr.get(0) if i in dropped
           else ir.AffineDimExpr.get(new_pos[i]) for i in range(n)]
    new_expr = expr.compose(ir.AffineMap.get(len(kept), 0, sel))
    new_map = ir.AffineMap.get(len(kept), 0, [new_expr])
    operands = list(apply_op.operands)
    new_operands = [operands[i] for i in kept]
    new_apply = ir.Operation.create(
        "affine.apply",
        results=[ir.IndexType.get(ctx)],
        operands=new_operands,
        attributes={"map": ir.AffineMapAttr.get(new_map)},
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(anchor_op),
    )
    return new_apply.results[0]


def _emit_compute(ctx, compute_node, tile_id, read_bufs, write_bufs):
    front = compute_node.operations[0]
    attrs = {
        ts.ATTR_TILE_ID: i64(tile_id),
        # int code (0 vector / 1 matmul / 2 preload) consumed by the C4 lowering;
        # maps directly to the Core compute-unit enum. Keep the readable name too.
        ts.ATTR_COMPUTE_TYPE: i32(int(compute_node.compute_type)),
        "compute_type_name": str_attr(_COMPUTE_TYPE_NAME[compute_node.compute_type]),
        # SRAM buffer ids read/written (sec 10 dataflow); the bridge builds the
        # dependency DAG by last-writer per buffer.
        ts.ATTR_READ_BUFS: i64_array(read_bufs),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    ir.Operation.create(
        ts.COMPUTE,
        results=[],
        operands=[],
        attributes=attrs,
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(front),
    )


# ---------------------------------------------------------------------------
# DCE
# ---------------------------------------------------------------------------
def _has_nonempty_region(op):
    for region in op.operation.regions:
        for b in region.blocks:
            if len(list(b.operations)) > 0:
                return True
    return False


def _results_unused(op):
    for r in op.operation.results:
        if len(list(r.uses)) > 0:
            return False
    return True


def _strip_loop_iter_args(block):
    """Drop loop-carried values (iter_args) from every affine.for/scf.for.

    The skeleton only needs the loop STRUCTURE (iteration counts) and the
    togsim.* markers -- not the data flowing through the loop. Reduction kernels
    carry a *vector* accumulator as an iter_arg; EmitC/C++ cannot represent a
    loop carrying a vector, so the trace .so emission fails. Since the trace is
    timing-only (values come from the recorded run), we rebuild each loop without
    iter_args: body uses of an iter_arg become its init value, the loop result
    becomes its init, and the now-orphaned accumulate ops are removed by _dce.
    """
    # Only strip a loop whose RESULTS are unused: a loop whose result still feeds a
    # kept op (an index accumulator behind a togsim.dma address) is left alone. Run
    # after _dce, so nested reductions free up inner results round by round.
    while True:
        tgt = None
        for op in walk_ops(block):
            n = op.operation.name
            if (n in ("affine.for", "scf.for") and len(op.operation.results) > 0
                    and _results_unused(op)):
                tgt = op
                break
        if tgt is None:
            return
        _rebuild_loop_no_iter(tgt)


def _rebuild_loop_no_iter(op):
    o = op.operation
    nres = len(o.results)
    n_in = len(o.operands)
    inits = [o.operands[n_in - nres + i] for i in range(nres)]
    keep_operands = [o.operands[i] for i in range(n_in - nres)]  # bound operands only
    old_block = o.regions[0].blocks[0]
    oargs = list(old_block.arguments)  # [iv, *iter_args]

    attrs = {na.name: na.attr for na in o.attributes}
    # affine.for tags its operand groups; zero the iter-arg group (last entry).
    if "operandSegmentSizes" in attrs:
        seg = [int(x) for x in str(attrs["operandSegmentSizes"]).split(":")[1].strip(" >").split(",")]
        seg[-1] = 0
        attrs["operandSegmentSizes"] = ir.Attribute.parse(
            "array<i32: " + ", ".join(str(s) for s in seg) + ">")

    loc = ir.Location.unknown(o.context)
    with loc:                                                  # default loc for new block args
        new = ir.Operation.create(o.name, results=[], operands=keep_operands,
                                  attributes=attrs, regions=1, loc=loc,
                                  ip=ir.InsertionPoint(o))
        nb = new.regions[0].blocks.append(oargs[0].type)      # block with the iv arg only

        oargs[0].replace_all_uses_with(nb.arguments[0])       # iv
        for ba, ini in zip(oargs[1:], inits):                 # iter-arg uses -> init
            ba.replace_all_uses_with(ini)
        for res, ini in zip(o.results, inits):                # loop result -> init
            res.replace_all_uses_with(ini)

        term_name = "affine.yield" if o.name == "affine.for" else "scf.yield"
        with ir.InsertionPoint(nb):
            ir.Operation.create(term_name, results=[], operands=[], loc=loc)
        new_term = list(nb.operations)[0]
        for bop in list(old_block.operations)[:-1]:           # move body (drop old yield)
            bop.operation.move_before(new_term)
        o.erase()


def _dce(block):
    """Erase non-kept ops with no used results, to a fixed point. Safe: an op
    with live SSA uses is never touched."""
    changed = True
    while changed:
        changed = False
        victims = []
        for op in walk_ops(block):
            name = op.operation.name
            if name in _KEEP:
                continue
            if _has_nonempty_region(op):
                continue
            if _results_unused(op):
                victims.append(op)
        for op in victims:
            try:
                op.operation.erase()
                changed = True
            except Exception:
                # Still referenced via something we will erase next round; retry.
                pass


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _collect_dma_nodes(builder):
    """Map op-identity -> DMA/DMAWait node, by walking the built tree."""
    by_op = {}
    seen = set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, (TOGDMANode, TOGDMAWaitNode)) and n.op is not None:
            by_op[id(n.op.operation)] = n
        for c in n.children:
            visit(c)

    for ln in builder.loop_nodes:
        visit(ln)
    for dn in getattr(builder, "dma_nodes", ()):   # DMAs outside any tile loop
        visit(dn)
    return by_op


class _BufferIds:
    """Assigns each SRAM buffer name a stable small int id, shared by DMA and
    compute so the bridge can match a reader to its buffer's writer (sec 10).
    The virtual SA_WEIGHTS buffer (preload -> matmul) is numbered here too, on
    first sight. `None` (a non-buffer base) is -1."""

    def __init__(self):
        self._ids = {}

    def of(self, name):
        if name is None:
            return -1
        return self._ids.setdefault(name, len(self._ids))


class _TagIds:
    """Identity of a DMA's tag memref -> stable small int, plus the SRAM buffer
    that tag's async DMA loads. An async dma and its memory_barrier (the original
    togsim.wait) share a tag memref; this assigns it a tag_id (so the runtime can
    pair them by the runtime tag slot) and remembers the loaded buffer so the
    barrier can release it to consumers. Pairing is by tag, never a static id."""

    def __init__(self):
        self._ids = {}   # tag value-key -> tag_id
        self._buf = {}   # tag value-key -> SRAM buffer id the dma loads

    def bind(self, key, buf):
        tag_id = self._ids.setdefault(key, len(self._ids))
        self._buf[key] = buf
        return tag_id

    def lookup(self, key):
        """(tag_id, buffer) for a tag memref, or None if no dma used it."""
        if key not in self._ids:
            return None
        return self._ids[key], self._buf[key]


def _emit_computes(ctx, builder, bufs):
    """Step 1: each compute node -> one togsim.compute carrying its tile_id and
    the ids of the SRAM buffers it reads/writes. Returns the count."""
    from . import dep_analysis as dep  # lazy: dep_analysis imports build_skeleton
    n = 0
    for tile_id, cn in enumerate(builder.compute_nodes):
        if not cn.operations:
            continue
        reads, writes = dep.compute_buffers(cn)
        _emit_compute(ctx, cn, tile_id,
                      sorted(bufs.of(b) for b in reads),
                      sorted(bufs.of(b) for b in writes))
        n += 1
    return n


def _transfer_fields(op):
    """Decode a `togsim.transfer`'s fixed operands by position.

    Layout (see mlir_codegen_backend.emit_transfer / lower_transfer_to_gemmini):
        operands: dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst
                  [, offset_spad]        # 8 or 9 operands
    Unlike the old `memref.dma_start`, dram/sram are FIXED (not direction-swapped):
    the DRAM side is always operand[0]/[1], the SRAM spad always operand[2], the
    runtime tag slot always operand[4] (tag memref) + operand[5] (tag_idx). The
    optional indirect-offset spad is operand[8]; its owning `memref.get_global`
    carries the offset symbol name in its "name" attribute (matching
    lower_transfer_to_gemmini's offset_sym derivation)."""
    operands = list(op.operands)
    offset = operands[8] if "indirect" in op.attributes else None
    return {
        "dram": operands[0], "dram_idx": operands[1],
        "sram": operands[2], "sram_idx": operands[3],
        "tag": operands[4], "tag_idx": operands[5],
        "dma_type": operands[6], "vst": operands[7],
        "offset": offset,
    }


def _emit_one_dma(ctx, op, node, builder, bufs, tags):
    """Rewrite one togsim.transfer as togsim.dma. A load reads DRAM and writes
    its SRAM spad; a store reads the spad and writes DRAM -- which sets the
    read/write buffer that drives the dependency edge (sec 10). The tag memref is
    bound to a tag_id (with its loaded buffer) so the paired memory_barrier finds
    it by the runtime tag slot."""
    from . import dep_analysis as dep  # lazy: dep_analysis imports build_skeleton
    f = _transfer_fields(op)
    # dram/sram are fixed operands now (not direction-swapped): the DRAM index is
    # always operand[1], the SRAM spad always operand[2]. Direction (read/write)
    # comes from the node (dma_kind attr / dma_type value).
    dram_index = f["dram_idx"]
    tag_index = f["tag_idx"]  # single runtime tag slot operand (%tag[%idx])
    spad_id = bufs.of(dep._global_of(f["sram"]))
    read_bufs = [spad_id] if node.is_write else []
    write_bufs = [] if node.is_write else [spad_id]
    if f["offset"] is not None:  # gather/scatter reads the offset spad -> dep on its build
        # the offset symbol name lives on the offset operand's owning get_global
        # ("name" attr), the same place lower_transfer_to_gemmini reads it.
        off_owner = f["offset"].owner
        off_sym = str(off_owner.attributes["name"]).strip('@" ')
        off_id = bufs.of(off_sym)
        if off_id not in read_bufs:
            read_bufs = read_bufs + [off_id]
    tag_id = tags.bind(_value_key(f["tag"]), spad_id)
    _emit_dma(ctx, node, tag_id, dram_index, tag_index, read_bufs, write_bufs)


def _emit_one_wait(ctx, op, tags):
    """Rewrite one togsim.wait as togsim.memory_barrier -- the explicit
    async-DMA sync already in the IR. Paired with its dma by the tag memref
    (tag_id) and the runtime tag index; carries the buffer the dma loaded.
    Returns True iff emitted (a wait whose tag no dma used is dropped)."""
    operands = list(op.operation.operands)
    tag = operands[0]
    tag_index = operands[1] if len(operands) >= 2 else None
    binding = tags.lookup(_value_key(tag))
    if binding is None:
        return False
    tag_id, buf = binding
    # honor lower_to_vcix's -1 accumulation marker: strip the reduction terms so
    # the barrier slot equals the subtile slot the paired async load wrote.
    tag_index = _strip_accum_terms(ctx, tag_index, op)
    _emit_memory_bar(ctx, op, tag_id, tag_index, [buf])
    return True


def _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs):
    """Step 2: rewrite togsim.transfer -> togsim.dma and togsim.wait ->
    togsim.memory_barrier in program order. An async dma and its barrier are
    paired by the RUNTIME tag slot (tag_id + tag index), not a compile-time id:
    one static dma op runs per loop iteration with a different `%tag[%idx]`, so
    only the runtime slot can pair iteration i's dma with iteration i's wait.
    Returns the original ops to erase and the (dma, wait) counts."""
    tags = _TagIds()
    originals = []
    n_dma = n_wait = 0
    for op in list(walk_ops(block)):
        name = op.operation.name
        if name == "togsim.transfer":
            node = dma_by_op.get(id(op.operation))
            if node is None:
                continue
            _emit_one_dma(ctx, op, node, builder, bufs, tags)
            originals.append(op)
            n_dma += 1
        elif name == "togsim.wait":
            if _emit_one_wait(ctx, op, tags):
                n_wait += 1
            originals.append(op)
    return originals, n_dma, n_wait


def build_skeleton(module):
    """Reduce `func.func @kernel` in `module` to the skeleton+API form, in place.

    Four steps: analyze the kernel into loop/compute/DMA nodes, emit a
    togsim.compute per compute node, rewrite the DMAs/waits to togsim.dma/wait,
    then DCE the leftover data computation. Returns a short text report (counts).
    """
    _reset_ids()
    builder = TogBuilder()
    _build(module, builder)  # populates loop/compute nodes + op back-pointers

    block = _kernel_block(module)
    if block is None:
        return "no @kernel found"
    ctx = module.context
    dma_by_op = _collect_dma_nodes(builder)
    bufs = _BufferIds()

    n_compute = _emit_computes(ctx, builder, bufs)
    originals, n_dma, n_wait = _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs)

    # erase the now-replaced originals (result-less -> safe), then strip the
    # leftover data computation.
    for op in originals:
        try:
            op.operation.erase()
        except Exception:
            pass
    _dce(block)                    # drop dead consumers (e.g. the result store) first,
    _strip_loop_iter_args(block)   # so a now-unused loop result lets us strip its iter_args
    _dce(block)                    # then clean the orphaned accumulate ops

    return ("skeleton: compute=%d dma=%d wait=%d (unpaired waits dropped)"
            % (n_compute, n_dma, n_wait))


def run(module, vectorlane=128):
    """passes/__init__ pass protocol entry (vectorlane unused; kept for parity)."""
    build_skeleton(module)


def run_skeleton(in_path, out_path=None):
    """Read post-vcix MLIR at `in_path`, reduce to skeleton+API, write it out.

    Requires the MLIR bindings.
    """
    if out_path is None:
        out_path = in_path
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        report = build_skeleton(module)
        with open(out_path, "w") as fh:
            fh.write(str(module))
    return report


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="build_skeleton.py")
    parser.add_argument("input")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv[1:])
    report = run_skeleton(args.input, args.out)
    import sys
    sys.stderr.write(report + "\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
