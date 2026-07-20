"""Reduce a >4D togsim.transfer to <=4D (drop unit dims / peel outer dims).

Both the Gemmini DMA descriptor and the TOGSim DMA model cap at 4 tile dims. A
logical tile can exceed 4D (e.g. pixel_shuffle splits two spatial axes -> a 5D
tile like [1,1,2,4,2]). lower_transfer_to_gemmini reduces >4D as part of emitting
Gemmini asm, but that runs only on the Spike/gem5 lowering path; the trace producer
(build_tog) reads togsim.transfer BEFORE that and would emit a >4D DMA that TOGSim
rejects ("issued tile is not supported format.. tile.size: 5").

This pass runs up front (POST_OPT, before build_tog) while KEEPING the op as
togsim.transfer, so both consumers see <=4D. It mirrors lower_transfer_to_gemmini's
two reductions: if the non-unit (effective) rank is <=4, collapse the unit dims
away (memref.collapse_shape); otherwise peel the outer effective dims into an
affine.for nest with the lane-banked physical SRAM offset. The innermost op is a
fresh <=4D togsim.transfer carrying the surviving dims' shape/strides/vlane axis
and the original dma_kind / masked / offset / subtile attributes.
"""

OP_NAME = "togsim.transfer"
MARKERS = (OP_NAME,)

from ._mlir_util import walk_ops
from .decompose_transfer import _int_array, _const_int, _squeeze_reassociation


def run(module, vectorlane=128, **_):
    """Reduce every >4D togsim.transfer in `module` to <=4D, in place. Context active.

    vectorlane (= systolic-array size / number of vector lanes) feeds the lane-banked
    physical SRAM offset in the peel, matching lower_transfer_to_gemmini.
    """
    from mlir.ir import (InsertionPoint, Operation, MemRefType, ArrayAttr,
                         IntegerAttr, IntegerType, IndexType, DenseI64ArrayAttr,
                         DenseI32ArrayAttr, StridedLayoutAttr, AffineMap, AffineMapAttr,
                         AffineExpr, BoolAttr)
    from mlir.dialects import affine
    i64 = IntegerType.get_signless(64)
    idx_ty = IndexType.get()

    def _arr(vals):
        return ArrayAttr.get([IntegerAttr.get(i64, int(v)) for v in vals])

    targets = []
    for region in module.operation.regions:
        for b in region.blocks:
            for op in walk_ops(b):
                if op.operation.name == OP_NAME:
                    targets.append(op.operation)

    for op in targets:
        op_operands = list(op.operands)
        dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst = op_operands[:8]
        sram_ty = MemRefType(sram.type)
        elem, space = sram_ty.element_type, sram_ty.memory_space
        tile_shape = list(sram_ty.shape)
        if len(tile_shape) <= 4:
            continue   # TOGSim / Gemmini accept <=4 raw tile dims -- nothing to do
        eff = [i for i, e in enumerate(tile_shape) if e > 1]

        has_indirect = "indirect" in op.attributes
        offset_operand = op_operands[8] if has_indirect else None
        vlane_axis = IntegerAttr(op.attributes["vlane_split_axis"]).value
        dram_stride = _int_array(op.attributes["dram_stride"])
        tile_stride = _int_array(op.attributes["tile_stride"])
        vlane_stride = _const_int(vst, 1)
        subtile = _int_array(op.attributes["subtile_size"]) if "subtile_size" in op.attributes else None
        if "masked_axes" in op.attributes:
            masked_axes = _int_array(op.attributes["masked_axes"])
            n_base = 9 if has_indirect else 8
            mvals = op_operands[n_base:]
            masked_pairs = [(masked_axes[i], mvals[2 * i], mvals[2 * i + 1])
                            for i in range(len(masked_axes))]
        else:
            masked_pairs = []

        def _emit_inner(sram_mem, sram_idx_val, dram_idx_val, new_vlane, dr, tl, st, mp):
            operands = [dram, dram_idx_val, sram_mem, sram_idx_val, tag, tag_idx, dma_type, vst]
            if has_indirect:
                operands.append(offset_operand)
            for _axis, lo, hi in mp:
                operands += [lo, hi]
            attrs = {
                "dma_kind": op.attributes["dma_kind"],
                "vlane_split_axis": IntegerAttr.get(i64, new_vlane),
                "dram_stride": _arr(dr),
                "tile_stride": _arr(tl),
                "padding": op.attributes["padding"],
            }
            if st is not None:
                attrs["subtile_size"] = _arr(st)
                if "async" in op.attributes:
                    attrs["async"] = op.attributes["async"]
            if mp:
                attrs["masked_axes"] = _arr([a for a, _lo, _hi in mp])
                attrs["masked_fill"] = op.attributes["masked_fill"]
            if has_indirect:
                attrs["indirect"] = op.attributes["indirect"]
                attrs["offset_stride"] = op.attributes["offset_stride"]
            if "accumulate" in op.attributes:
                attrs["accumulate"] = op.attributes["accumulate"]
            if "acc_float" in op.attributes:
                attrs["acc_float"] = op.attributes["acc_float"]
            Operation.create(OP_NAME, results=[], operands=operands, attributes=attrs)

        # <=4 effective dims: collapse the unit dims so the raw rank reaches <=4.
        if len(eff) <= 4:
            groups, target = _squeeze_reassociation(tile_shape)
            reassoc = ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(i64, d) for d in g]) for g in groups])
            collapsed_ty = MemRefType.get(target, elem, memory_space=space)
            keep = [next((d for d in g if tile_shape[d] > 1), g[-1]) for g in groups]
            dr = [dram_stride[i] for i in keep]
            tl = [tile_stride[i] for i in keep]
            st = [subtile[i] for i in keep] if subtile is not None else None
            mp = [(next(gi for gi, g in enumerate(groups) if d in g), lo, hi)
                  for d, lo, hi in masked_pairs]
            new_vlane = next(gi for gi, g in enumerate(groups) if vlane_axis in g)
            with InsertionPoint(op):
                sram_c = Operation.create(
                    "memref.collapse_shape", results=[collapsed_ty], operands=[sram],
                    attributes={"reassociation": reassoc}).results[0]
                _emit_inner(sram_c, sram_idx, dram_idx, new_vlane, dr, tl, st, mp)
            op.erase()
            continue

        # >4 effective dims: peel the outer ones into an affine.for nest, keep the
        # innermost 4. The SRAM slice offset is the lane-banked physical offset,
        # delivered as sram_idx; the DRAM offset folds into dram_idx.
        peeled, inner = eff[:-4], eff[-4:]
        ndim = len(tile_shape)
        inner_shape = [tile_shape[d] for d in inner]
        inner_strides = [tile_stride[d] for d in inner]
        dr = [dram_stride[d] for d in inner]
        tl = [tile_stride[d] for d in inner]
        st = [subtile[d] for d in inner] if subtile is not None else None
        if any(d in peeled for d, _lo, _hi in masked_pairs):
            raise NotImplementedError("masked-DMA clamp on a peeled (outer-loop) axis")
        mp = [(inner.index(d), lo, hi) for d, lo, hi in masked_pairs if d in inner]
        if vlane_axis in inner:
            new_vlane = inner.index(vlane_axis)
        elif vlane_axis in peeled:
            raise NotImplementedError(
                f"vlane split axis {vlane_axis} peeled into the outer loop nest")
        else:
            new_vlane = 0

        split_extent = tile_shape[vlane_axis]
        nr_outerloop = max(
            (split_extent + vectorlane * vlane_stride - 1) // (vectorlane * vlane_stride), 1)
        new_size = nr_outerloop * vlane_stride
        target_stride = tile_stride[vlane_axis]

        def _phys(d):
            s = tile_stride[d]
            return s // split_extent * new_size if s > target_stride else s

        static_sizes = [1] * ndim
        for d in inner:
            static_sizes[d] = tile_shape[d]
        res_ty = MemRefType.get(
            inner_shape, elem,
            layout=StridedLayoutAttr.get(0, inner_strides), memory_space=space)

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
                            "operandSegmentSizes": DenseI32ArrayAttr.get([1, 0, 0, 0])}
            ).results[0]
            sram_expr = AffineExpr.get_dim(0) * _phys(peeled[0])
            for k in range(1, npeel):
                sram_expr = sram_expr + AffineExpr.get_dim(k) * _phys(peeled[k])
            sram_off_val = Operation.create(
                "affine.apply", results=[idx_ty], operands=list(ivs),
                attributes={"map": AffineMapAttr.get(AffineMap.get(npeel, 0, [sram_expr]))}
            ).results[0]
            dram_expr = AffineExpr.get_dim(0)
            for k in range(npeel):
                dram_expr = dram_expr + AffineExpr.get_dim(k + 1) * dram_stride[peeled[k]]
            dram_idx_val = Operation.create(
                "affine.apply", results=[idx_ty], operands=[dram_idx, *ivs],
                attributes={"map": AffineMapAttr.get(AffineMap.get(npeel + 1, 0, [dram_expr]))}
            ).results[0]
            _emit_inner(sub, sram_off_val, dram_idx_val, new_vlane, dr, tl, st, mp)
        op.erase()


def peel_text(text):
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
    out = peel_text(open(sys.argv[1]).read())
    (open(sys.argv[2], "w").write(out) if len(sys.argv) > 2 else sys.stdout.write(out))
