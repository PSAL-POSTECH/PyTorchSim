"""Python out-of-line MLIR pass: decompose togsim.transfer -> <=4D memref.dma_start.

A togsim.transfer carries a per-axis affine DMA whose descriptor rank may exceed
the 4D Gemmini limit. This pass is a **pure mechanical rank peel** of that
already-affine access (see docs/dma-transfer-lowering.md, "aligned-only peel"):

  - drop unit (extent-1) tile dims: they contribute no descriptor axis;
  - if the remaining (effective) rank <= 4 -> emit one customized
    memref.dma_start, reusing the transfer's operands (fast path);
  - if effective rank > 4 -> peel the outer dims into a loop, adjusting the
    base index by stride*iv per iteration, inner descriptor <=4D.

It does NO floor/mod linearization (aligned split happens upstream at the
scheduling layer) and NO relayout (misaligned access is copy-inserted at the
graph level). A transfer whose access is not per-axis affine is a contract
violation -- but by construction codegen only emits affine transfers.

togsim.transfer operands (see emit_transfer):
    (dram, dram_idx, sram, sram_idx, tag, dma_type, vlane_split_axis, vlane_stride)
attrs: dma_kind ("MVIN"/"MVOUT"), dram_stride[], tile_stride[], padding.

memref.dma_start (customized) operands:
    src[idx], dst[idx], dma_type, tag[idx], vlane_split_axis, vlane_stride
    : src_memref, dst_memref, memref<1xi32> {dram_stride, sram_stride, padding}

Pass interface (passes/__init__.py): MARKERS + run(module).
"""

OP_NAME = "togsim.transfer"
MARKERS = (OP_NAME,)


def _iter_ops(block):
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _iter_ops(b)


def _int_array(attr):
    from mlir.ir import ArrayAttr, IntegerAttr
    return [IntegerAttr(a).value for a in ArrayAttr(attr)]


def _squeeze_reassociation(shape):
    """Group source dims so each group's product is one effective (non-unit) dim;
    unit dims attach to a neighbor. Returns (groups, target_shape)."""
    groups, cur = [], []
    for i, e in enumerate(shape):
        cur.append(i)
        if e > 1:
            groups.append(cur)
            cur = []
    if cur:                                   # trailing unit dims
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)                # all-ones -> single dim of size 1
    import math
    target = [math.prod(shape[d] for d in g) for g in groups]
    return groups, target


def run(module):
    """Lower every togsim.transfer in `module`, in place. Context must be active."""
    from mlir.ir import (InsertionPoint, Operation, MemRefType, ArrayAttr,
                         IntegerAttr, IntegerType, IndexType)
    i64 = IntegerType.get_signless(64)

    targets = []
    for region in module.operation.regions:
        for b in region.blocks:
            for op in _iter_ops(b):
                if op.operation.name == OP_NAME:
                    targets.append(op.operation)

    for op in targets:
        dram, dram_idx, sram, sram_idx, tag, dma_type, vst = op.operands
        kind = op.attributes["dma_kind"].value          # StringAttr -> "MVIN"/"MVOUT"
        vlane_axis = IntegerAttr(op.attributes["vlane_split_axis"]).value
        dram_stride = _int_array(op.attributes["dram_stride"])
        tile_stride = _int_array(op.attributes["tile_stride"])
        padding = op.attributes["padding"]

        sram_ty = MemRefType(sram.type)
        tile_shape = list(sram_ty.shape)
        # effective (non-unit) dims carry the descriptor; unit dims drop out.
        eff = [i for i, e in enumerate(tile_shape) if e > 1]

        if len(eff) > 4:
            raise NotImplementedError(
                f"{OP_NAME}: effective rank {len(eff)} > 4 needs the peel loop "
                "(not yet implemented); only unit-dim drop / <=4D is handled")

        # The customized memref.dma_start convention: the SRAM memref rank == number
        # of SRAM indices == len(sram_stride). To reach a <=4D descriptor we collapse
        # the unit tile dims away, then index the collapsed memref with one base per
        # remaining dim. DRAM stays flat rank-1 (its N-D structure is in dram_stride).
        groups, target = _squeeze_reassociation(tile_shape)
        rank = len(target)
        reassoc = ArrayAttr.get(
            [ArrayAttr.get([IntegerAttr.get(i64, d) for d in g]) for g in groups])
        collapsed_ty = MemRefType.get(target, sram_ty.element_type,
                                      memory_space=sram_ty.memory_space)

        # strides for the surviving (effective) dims, aligned with `target`/`groups`.
        keep = [g[-1] for g in groups]                  # the non-unit dim in each group
        inner_dram = ArrayAttr.get([IntegerAttr.get(i64, dram_stride[i]) for i in keep])
        inner_tile = ArrayAttr.get([IntegerAttr.get(i64, tile_stride[i]) for i in keep])

        # Remap the vlane axis from the original tile-dim index to the collapsed-dim
        # index (the group that contains it), then materialize the new const.
        new_vlane_axis = next(gi for gi, g in enumerate(groups) if vlane_axis in g)
        idx_ty = IndexType.get()

        with InsertionPoint(op):
            vsa = Operation.create(
                "arith.constant", results=[idx_ty],
                attributes={"value": IntegerAttr.get(idx_ty, new_vlane_axis)}).results[0]
            sram_c = Operation.create(
                "memref.collapse_shape", results=[collapsed_ty], operands=[sram],
                attributes={"reassociation": reassoc}).results[0]
            sram_indices = [sram_idx] * rank
            if kind == "MVIN":
                operands = [dram, dram_idx, sram_c, *sram_indices,
                            dma_type, tag, sram_idx, vsa, vst]
            else:
                operands = [sram_c, *sram_indices, dram, dram_idx,
                            dma_type, tag, sram_idx, vsa, vst]
            Operation.create(
                "memref.dma_start", results=[], operands=operands,
                attributes={"dram_stride": inner_dram, "sram_stride": inner_tile,
                            "padding": padding})
        op.erase()


def lower_text(text: str) -> str:
    """Parse `text`, run this pass, return the printed module. CLI/testing helper."""
    if OP_NAME not in text:
        return text
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(text)
        run(m)
        return str(m)


if __name__ == "__main__":
    import sys
    out = lower_text(open(sys.argv[1]).read())
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
    else:
        sys.stdout.write(out)
