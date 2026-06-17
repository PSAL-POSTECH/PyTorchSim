"""Python out-of-line MLIR pass: decompose togsim.transfer -> <=4D memref.dma_start.

A togsim.transfer carries a per-axis affine DMA whose descriptor rank may exceed
the 4D Gemmini limit. This pass lowers it to <=4D customized memref.dma_start
(see docs/dma-transfer-lowering.md):

  - drop unit (extent-1) tile dims: they contribute no descriptor axis;
  - if the remaining (effective) rank <= 4 -> emit one customized
    memref.dma_start, reusing the transfer's operands (fast path);
  - if effective rank > 4 -> wrap the outer dims in an affine.for nest and emit
    one <=4D memref.dma_start in the body, mirroring the C++ -dma-fine-grained
    subtile loop. The slice DRAM/SRAM offsets are affine.apply over the loop vars;
    the SRAM offset is the lane-banked physical offset (split-outer dims rescaled
    by the lane coeff) delivered as the last SRAM index operand.

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


def _const_int(value, default=None):
    """Read an arith.constant index/integer operand's value, else `default`."""
    from mlir.ir import IntegerAttr
    try:
        return IntegerAttr(value.owner.attributes["value"]).value
    except Exception:
        return default


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


def run(module, vectorlane=128, **_):
    """Lower every togsim.transfer in `module`, in place. Context must be active.

    vectorlane (= systolic-array size / number of vector lanes) feeds the lane-banked
    physical SRAM offset in the >4D peel, matching -dma-fine-grained's
    systolic-array-size option.
    """
    from mlir.ir import (InsertionPoint, Operation, MemRefType, ArrayAttr,
                         IntegerAttr, IntegerType, IndexType, DenseI64ArrayAttr,
                         DenseI32ArrayAttr, StridedLayoutAttr, AffineMap, AffineMapAttr,
                         AffineExpr, BoolAttr)
    from mlir.dialects import affine
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
        vlane_stride = _const_int(vst, 1)
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

        # Peel path: >4 effective dims. Wrap the outer (len-4) effective dims in an
        # affine.for nest (one loop per outer dim, marked inner_loop so build_tog/TOG
        # registers the induction var) and emit a single <=4D memref.dma_start in the
        # innermost body -- mirroring the C++ -dma-fine-grained subtile loop.
        #
        # The slice SRAM offset is the PHYSICAL lane-banked offset: dims outer than the
        # vlane axis are rescaled by the lane coeff (stride/old_size*new_size, the MVIN
        # block_stride / buildSramAffineMap rule). It is delivered as the last SRAM index
        # operand (row-major stride 1), NOT a subview offset -- the gemmini lowering reads
        # the spad base via extract_aligned_pointer_as_index, which strips the subview
        # offset, so the slice must be selected through the index. The DRAM offset is the
        # flat contiguous offset, folded with the original dram_idx into one affine.apply
        # (an arith.addi would be opaque to processDramIndices -- #258); the affine.for
        # induction vars feed both maps so TOG reads the loop indices through them.
        peeled, inner = eff[:-4], eff[-4:]
        ndim = len(tile_shape)
        inner_shape = [tile_shape[d] for d in inner]
        inner_strides = [tile_stride[d] for d in inner]
        dr_attr = ArrayAttr.get([IntegerAttr.get(i64, dram_stride[d]) for d in inner])
        tl_attr = ArrayAttr.get([IntegerAttr.get(i64, tile_stride[d]) for d in inner])
        # the vlane axis must survive into the inner descriptor (it is the lane dim).
        new_vlane = inner.index(vlane_axis) if vlane_axis in inner else 0

        # Lane-banked physical stride for split-outer dims (vlane_stride defaults to 1).
        split_extent = tile_shape[vlane_axis]
        nr_outerloop = max(
            (split_extent + vectorlane * vlane_stride - 1) // (vectorlane * vlane_stride), 1)
        new_size = nr_outerloop * vlane_stride
        target_stride = tile_stride[vlane_axis]

        def _phys(d):
            s = tile_stride[d]
            return s // split_extent * new_size if s > target_stride else s

        # subview to the inner <=4D block at the buffer start (offset 0); slice selection
        # is done through the SRAM index, so the StridedLayout offset stays 0.
        static_sizes = [1] * ndim
        for d in inner:
            static_sizes[d] = tile_shape[d]
        res_ty = MemRefType.get(
            inner_shape, elem,
            layout=StridedLayoutAttr.get(0, inner_strides), memory_space=space)

        # affine.for nest over the peeled (outer) dims.
        cur_ip = InsertionPoint(op)
        ivs = []
        for d in peeled:
            floop = affine.AffineForOp(0, tile_shape[d], 1, ip=cur_ip)
            floop.operation.attributes["inner_loop"] = BoolAttr.get(True)
            ivs.append(floop.induction_variable)
            with InsertionPoint(floop.body):
                affine.AffineYieldOp([])
            cur_ip = InsertionPoint.at_block_terminator(floop.body)

        npeel = len(peeled)
        with cur_ip:
            sub = Operation.create(
                "memref.subview", results=[res_ty], operands=[sram],
                attributes={"static_offsets": DenseI64ArrayAttr.get([0] * ndim),
                            "static_sizes": DenseI64ArrayAttr.get(static_sizes),
                            "static_strides": DenseI64ArrayAttr.get([1] * ndim),
                            # i32 [source, offsets, sizes, strides] dynamic-operand counts;
                            # all static -> source only. i64 silently zeroes and fails verify.
                            "operandSegmentSizes": DenseI32ArrayAttr.get([1, 0, 0, 0])}
            ).results[0]
            # physical SRAM offset = sum_k iv_k * phys_stride(peeled[k])
            sram_expr = AffineExpr.get_dim(0) * _phys(peeled[0])
            for k in range(1, npeel):
                sram_expr = sram_expr + AffineExpr.get_dim(k) * _phys(peeled[k])
            sram_off_val = Operation.create(
                "affine.apply", results=[idx_ty], operands=list(ivs),
                attributes={"map": AffineMapAttr.get(AffineMap.get(npeel, 0, [sram_expr]))}
            ).results[0]
            # DRAM index = orig dram_idx + sum_k iv_k * dram_stride(peeled[k])
            dram_expr = AffineExpr.get_dim(0)
            for k in range(npeel):
                dram_expr = dram_expr + AffineExpr.get_dim(k + 1) * dram_stride[peeled[k]]
            dram_idx_val = Operation.create(
                "affine.apply", results=[idx_ty], operands=[dram_idx, *ivs],
                attributes={"map": AffineMapAttr.get(AffineMap.get(npeel + 1, 0, [dram_expr]))}
            ).results[0]
            zero = _const(0)
            _emit(sub, [zero, zero, zero, sram_off_val], dram_idx_val, new_vlane,
                  dr_attr, tl_attr)
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
