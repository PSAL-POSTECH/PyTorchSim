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
    import itertools
    from mlir.ir import (InsertionPoint, Operation, MemRefType, ArrayAttr,
                         IntegerAttr, IntegerType, IndexType, DenseI64ArrayAttr,
                         DenseI32ArrayAttr, StridedLayoutAttr)
    i64 = IntegerType.get_signless(64)
    idx_ty = IndexType.get()

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
        elem, space = sram_ty.element_type, sram_ty.memory_space
        tile_shape = list(sram_ty.shape)
        # effective (non-unit) dims carry the descriptor; unit dims drop out.
        eff = [i for i, e in enumerate(tile_shape) if e > 1]

        def _const(v):
            return Operation.create(
                "arith.constant", results=[idx_ty],
                attributes={"value": IntegerAttr.get(idx_ty, v)}).results[0]

        def _emit(sram_mem, sram_indices, dram_idx_val, vsa_val, dr_attr, tl_attr):
            vsa = _const(vsa_val)
            if kind == "MVIN":
                operands = [dram, dram_idx_val, sram_mem, *sram_indices,
                            dma_type, tag, sram_idx, vsa, vst]
            else:
                operands = [sram_mem, *sram_indices, dram, dram_idx_val,
                            dma_type, tag, sram_idx, vsa, vst]
            Operation.create(
                "memref.dma_start", results=[], operands=operands,
                attributes={"dram_stride": dr_attr, "sram_stride": tl_attr,
                            "padding": padding})

        if len(eff) <= 4:
            # Fast path: drop unit dims so the descriptor reaches <=4D. The customized
            # dma_start convention requires SRAM rank == #indices == len(sram_stride),
            # so collapse the unit tile dims away. DRAM stays flat rank-1 (its N-D
            # structure is in dram_stride).
            groups, target = _squeeze_reassociation(tile_shape)
            reassoc = ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(i64, d) for d in g]) for g in groups])
            collapsed_ty = MemRefType.get(target, elem, memory_space=space)
            keep = [g[-1] for g in groups]              # the non-unit dim in each group
            dr_attr = ArrayAttr.get([IntegerAttr.get(i64, dram_stride[i]) for i in keep])
            tl_attr = ArrayAttr.get([IntegerAttr.get(i64, tile_stride[i]) for i in keep])
            # Remap vlane axis to the collapsed-dim index (the group containing it).
            new_vlane = next(gi for gi, g in enumerate(groups) if vlane_axis in g)
            with InsertionPoint(op):
                sram_c = Operation.create(
                    "memref.collapse_shape", results=[collapsed_ty], operands=[sram],
                    attributes={"reassociation": reassoc}).results[0]
                _emit(sram_c, [sram_idx] * len(target), dram_idx, new_vlane,
                      dr_attr, tl_attr)
            op.erase()
            continue

        # Peel path: >4 effective dims. Keep the inner 4 as the <=4D descriptor and
        # peel the outer (len-4) effective dims into a fully-unrolled set of slices
        # (one descriptor per outer index combo; base advances by stride*idx). The
        # SRAM slice is a rank-reduced memref.subview at the slice offset; DRAM base
        # is dram_idx + constant. Unrolling (vs scf.for) keeps the slice offsets
        # static so no per-iteration index arithmetic on the SRAM side is needed.
        #
        # NOTE: currently unreachable -- init_tile_size caps non-unit tile dims at 3,
        # so eff <= 3 in practice. Implemented for completeness / future tilings and
        # validated only in isolation (passes/decompose_transfer.py CLI / lower_text).
        peeled, inner = eff[:-4], eff[-4:]
        ndim = len(tile_shape)
        inner_shape = [tile_shape[d] for d in inner]
        inner_strides = [tile_stride[d] for d in inner]
        dr_attr = ArrayAttr.get([IntegerAttr.get(i64, dram_stride[d]) for d in inner])
        tl_attr = ArrayAttr.get([IntegerAttr.get(i64, tile_stride[d]) for d in inner])
        # the vlane axis must survive into the inner descriptor (it is the lane dim).
        new_vlane = inner.index(vlane_axis) if vlane_axis in inner else 0
        for combo in itertools.product(*[range(tile_shape[d]) for d in peeled]):
            static_offsets = [0] * ndim
            static_sizes = [1] * ndim
            for k, d in enumerate(peeled):
                static_offsets[d] = combo[k]
            for d in inner:
                static_sizes[d] = tile_shape[d]
            sram_off = sum(combo[k] * tile_stride[peeled[k]] for k in range(len(peeled)))
            dram_off = sum(combo[k] * dram_stride[peeled[k]] for k in range(len(peeled)))
            res_ty = MemRefType.get(
                inner_shape, elem,
                layout=StridedLayoutAttr.get(sram_off, inner_strides), memory_space=space)
            with InsertionPoint(op):
                sub = Operation.create(
                    "memref.subview", results=[res_ty], operands=[sram],
                    attributes={"static_offsets": DenseI64ArrayAttr.get(static_offsets),
                                "static_sizes": DenseI64ArrayAttr.get(static_sizes),
                                "static_strides": DenseI64ArrayAttr.get([1] * ndim),
                                # operandSegmentSizes is an i32 property: [source, offsets,
                                # sizes, strides] dynamic-operand counts. All static here ->
                                # only the source operand. Must be i32, not i64 (i64 silently
                                # zeroes to [0,0,0,0] and fails verification).
                                "operandSegmentSizes": DenseI32ArrayAttr.get([1, 0, 0, 0])}
                ).results[0]
                dram_idx_val = dram_idx if dram_off == 0 else Operation.create(
                    "arith.addi", results=[idx_ty],
                    operands=[dram_idx, _const(dram_off)]).results[0]
                _emit(sub, [sram_idx] * 4, dram_idx_val, new_vlane, dr_attr, tl_attr)
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
