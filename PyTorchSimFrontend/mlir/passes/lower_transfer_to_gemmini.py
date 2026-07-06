"""Lower togsim.transfer DIRECTLY to Gemmini RISC-V inline asm (no memref.dma_start).

Merges decompose_transfer (the <=4D Gemmini-limit handling: drop unit dims /
collapse / >4D affine.for peel with lane-banked SRAM offset) with
lower_dma_to_gemmini (the CONFIG/CONFIG2/CONFIG3/[CONFIG4]/MVIN|MVOUT asm emission).
togsim.transfer is unregistered so it carries every runtime descriptor as an
operand -- including the future masked-clamp low/high vectors -- which a registered
memref.dma_start cannot.

timing=False: emit the gemmini asm. timing=True: erase the transfer (the TOG carries
DMA timing; the cycle binary needs no asm).
"""

OP_NAME = "togsim.transfer"
WAIT_NAME = "togsim.wait"
MARKERS = (OP_NAME, WAIT_NAME)

from ._mlir_util import walk_ops
from .lower_dma_to_gemmini import _i64_signed, _row_major_strides, _elem_bytes, _asm, CONSTRAINTS
from .decompose_transfer import _int_array, _const_int, _squeeze_reassociation

CONFIG, CONFIG2, CONFIG3, CONFIG4, CONFIG_DESC = 0, 4, 5, 6, 7
MVIN, MVIN2, MVIN3, MVOUT = 2, 1, 14, 3
DESC_SLOTS = 18   # 144-byte DMA descriptor as 18 x i64 (see project-dma-descriptor)
CONFIG_TYPE = {MVIN: 0, MVIN2: 1, MVIN3: 2, MVOUT: 3}
MAX_TENSOR_DIM = 4


def run(module, timing=False, vectorlane=128, **_):
    from mlir.ir import (InsertionPoint, Operation, MemRefType, ArrayAttr,
                         IntegerAttr, IntegerType, IndexType, DenseI64ArrayAttr,
                         DenseI32ArrayAttr, StridedLayoutAttr, AffineMap, AffineMapAttr,
                         AffineExpr, BoolAttr, FlatSymbolRefAttr, TypeAttr,
                         DenseElementsAttr, RankedTensorType, StringAttr, UnitAttr)
    from mlir.dialects import affine, llvm, arith, memref
    i64 = IntegerType.get_signless(64)
    idx_ty = IndexType.get()
    module_top = module.operation.regions[0].blocks[0]

    sym2type = {}
    for g in module_top.operations:
        if g.operation.name == "memref.global":
            sym2type[g.attributes["sym_name"].value] = MemRefType(TypeAttr(g.attributes["type"]).value)

    desc_ty = MemRefType.get([DESC_SLOTS], i64)
    desc_globals = {}   # slots tuple -> global sym name (dedup)
    desc_counter = [0]

    def _pack_desc(shape4, dram4, spad4, elem_bytes, vlstride, vsa4, cfg_type, indirect,
                   ind_stride, ind_esize):
        # 18 i64 slots matching the C dma_descriptor byte layout (little-endian).
        lo = [0, 0, 0, 0]
        hi = list(shape4)
        def s2(a, b): return (a & 0xFFFFFFFF) | ((b & 0xFFFFFFFF) << 32)
        m = 0xFFFFFFFFFFFFFFFF
        slots = [
            s2(shape4[0], shape4[1]), s2(shape4[2], shape4[3]),
            s2(lo[0], lo[1]), s2(lo[2], lo[3]),
            s2(hi[0], hi[1]), s2(hi[2], hi[3]),
            dram4[0] & m, dram4[1] & m, dram4[2] & m, dram4[3] & m,
            spad4[0] & m, spad4[1] & m, spad4[2] & m, spad4[3] & m,
            (elem_bytes & 0xFFFF) | ((vlstride & 0xFFFF) << 16) | ((vsa4 & 0xFF) << 32)
                | ((cfg_type & 0xFF) << 40) | (((1 if indirect else 0) & 0xFFFF) << 48),
            0,                                                    # +120 indirect_addr (runtime)
            (ind_stride & 0xFFFF) | ((ind_esize & 0xFFFF) << 16),  # +128
            0,                                                    # +136 fill (runtime/step3)
        ]
        return slots

    def _desc_global(slots):
        key = tuple(slots)
        if key not in desc_globals:
            name = f"dma_desc_{desc_counter[0]}"
            desc_counter[0] += 1
            import numpy as np
            init = DenseElementsAttr.get(
                np.array([_i64_signed(v) for v in slots], dtype=np.int64),
                type=RankedTensorType.get([DESC_SLOTS], i64))
            with InsertionPoint.at_block_begin(module_top):
                Operation.create("memref.global", results=[], operands=[], attributes={
                    "sym_name": StringAttr.get(name),
                    "sym_visibility": StringAttr.get("private"),
                    "type": TypeAttr.get(desc_ty),
                    "initial_value": init})
            desc_globals[key] = name
        return desc_globals[key]

    def desc_ptr_and_store_indirect(slots, ind_addr_i64):
        # get_global -> byte pointer; for indirect, store the runtime addr into slot 15.
        name = _desc_global(slots)
        g = memref.GetGlobalOp(desc_ty, name).result
        if ind_addr_i64 is not None:
            memref.StoreOp(ind_addr_i64, g, [arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 15)).result])
        base = memref.ExtractAlignedPointerAsIndexOp(g).result
        return arith.IndexCastOp(i64, base).result

    def i64_const(value):
        return arith.ConstantOp(i64, IntegerAttr.get(i64, _i64_signed(value))).result

    def asm(func7, rs1, rs2):
        llvm.InlineAsmOp(None, [rs1, rs2], _asm(func7), CONSTRAINTS,
                         has_side_effects=True, asm_dialect=0)

    def elem_addr_i64(memref_val, indices, mtype, elem_bytes):
        base = memref.ExtractAlignedPointerAsIndexOp(memref_val).result
        strides = _row_major_strides(list(mtype.shape))
        off = None
        for k, ival in enumerate(indices):
            if strides[k] == 0:
                continue
            term = ival
            if strides[k] != 1:
                term = arith.MulIOp(ival, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, strides[k])).result).result
            off = term if off is None else arith.AddIOp(off, term).result
        if off is not None:
            byte = arith.MulIOp(off, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, elem_bytes)).result).result
            base = arith.AddIOp(base, byte).result
        return arith.IndexCastOp(i64, base).result

    targets, waits = [], []
    for region in module.operation.regions:
        for b in region.blocks:
            for op in walk_ops(b):
                if op.operation.name == OP_NAME:
                    targets.append(op.operation)
                elif op.operation.name == WAIT_NAME:
                    waits.append(op.operation)

    for op in waits:        # togsim.wait: erase in both modes (the barrier is a sync marker)
        op.erase()

    for op in targets:
        op_operands = list(op.operands)
        dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst = op_operands[:8]
        offset_sym = (op_operands[8].owner.attributes["name"] if "indirect" in op.attributes else None)
        kind = op.attributes["dma_kind"].value
        dma_type_val = _const_int(dma_type)          # MVIN(2)/MVIN2(1)/MVIN3(14)/MVOUT(3)
        is_mvin = dma_type_val in (MVIN, MVIN2, MVIN3)
        vlane_axis = IntegerAttr(op.attributes["vlane_split_axis"]).value
        dram_stride = _int_array(op.attributes["dram_stride"])
        tile_stride = _int_array(op.attributes["tile_stride"])
        vlane_stride = _const_int(vst, 1)
        try:
            subtile = _int_array(op.attributes["subtile_size"])
        except KeyError:
            subtile = None

        if timing:
            op.erase()
            continue

        sram_ty = MemRefType(sram.type)
        elem, space = sram_ty.element_type, sram_ty.memory_space
        elem_bytes = _elem_bytes(sram_ty.element_type)
        dram_ty = MemRefType(dram.type)
        tile_shape = list(sram_ty.shape)
        eff = [i for i, e in enumerate(tile_shape) if e > 1]
        indirect = offset_sym is not None

        def _const(v):
            return arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, v)).result

        def _emit_asm(sram_mem, sram_indices, dram_idx_val, vsa_val, desc_shape,
                      desc_dram_strides, desc_spad_strides, subtile_shape):
            cfg_shape = subtile_shape if subtile_shape is not None else desc_shape
            expand = MAX_TENSOR_DIM - len(cfg_shape)
            shape4 = [1] * expand + list(cfg_shape)
            dram4 = [0] * expand + list(desc_dram_strides)
            spad4 = [0] * expand + list(desc_spad_strides)
            vsa4 = vsa_val + expand
            config_type = CONFIG_TYPE[dma_type_val]
            sram_c = MemRefType(sram_mem.type)
            dram_addr = elem_addr_i64(dram, [dram_idx_val], dram_ty, elem_bytes)
            spad_addr = elem_addr_i64(sram_mem, sram_indices, sram_c, elem_bytes)
            ind_addr = None
            ind_stride = ind_esize = 0
            if indirect:
                sym = FlatSymbolRefAttr(offset_sym).value if not isinstance(offset_sym, str) else offset_sym
                off_ty = sym2type[sym]
                ind_base = memref.ExtractAlignedPointerAsIndexOp(memref.GetGlobalOp(off_ty, sym).result).result
                ind_addr = arith.IndexCastOp(i64, ind_base).result
                ind_esize = _elem_bytes(off_ty.element_type)
                ind_stride = IntegerAttr(op.attributes["offset_stride"]).value
            slots = _pack_desc(shape4, dram4, spad4, elem_bytes, vlane_stride, vsa4,
                               config_type, indirect, ind_stride, ind_esize)
            desc_ptr = desc_ptr_and_store_indirect(slots, ind_addr)
            asm(CONFIG_DESC, desc_ptr, i64_const(0))
            asm(dma_type_val, dram_addr, spad_addr)

        if offset_sym is not None:
            offset_sym = FlatSymbolRefAttr(offset_sym).value if not isinstance(offset_sym, str) else offset_sym

        if len(tile_shape) <= 4:
            with InsertionPoint(op):
                # sram offset is linear (row-major stride 1) -> last index only; others 0.
                sidx = [_const(0)] * (len(tile_shape) - 1) + [sram_idx]
                _emit_asm(sram, sidx, dram_idx, vlane_axis,
                          tile_shape, dram_stride, tile_stride, subtile)
            op.erase()
            continue

        if len(eff) <= 4:
            groups, target = _squeeze_reassociation(tile_shape)
            reassoc = ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(i64, d) for d in g]) for g in groups])
            collapsed_ty = MemRefType.get(target, elem, memory_space=space)
            keep = [next((d for d in g if tile_shape[d] > 1), g[-1]) for g in groups]
            dr = [dram_stride[i] for i in keep]
            tl = [tile_stride[i] for i in keep]
            st = [subtile[i] for i in keep] if subtile is not None else None
            new_vlane = next(gi for gi, g in enumerate(groups) if vlane_axis in g)
            with InsertionPoint(op):
                sram_c = Operation.create(
                    "memref.collapse_shape", results=[collapsed_ty], operands=[sram],
                    attributes={"reassociation": reassoc}).results[0]
                sidx = [_const(0)] * (len(target) - 1) + [sram_idx]
                _emit_asm(sram_c, sidx, dram_idx, new_vlane, target, dr, tl, st)
            op.erase()
            continue

        # >4 effective dims: affine.for peel (mirrors decompose_transfer peel path)
        peeled, inner = eff[:-4], eff[-4:]
        ndim = len(tile_shape)
        inner_shape = [tile_shape[d] for d in inner]
        inner_strides = [tile_stride[d] for d in inner]
        dr = [dram_stride[d] for d in inner]
        tl = [tile_stride[d] for d in inner]
        st = [subtile[d] for d in inner] if subtile is not None else None
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
            zero = _const(0)
            _emit_asm(sub, [zero, zero, zero, sram_off_val], dram_idx_val, new_vlane,
                      inner_shape, dr, tl, st)
        op.erase()


def lower_text(text, timing=False):
    if OP_NAME not in text:
        return text
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(text)
        run(m, timing=timing)
        return str(m)


if __name__ == "__main__":
    import sys
    out = lower_text(open(sys.argv[1]).read())
    (open(sys.argv[2], "w").write(out) if len(sys.argv) > 2 else sys.stdout.write(out))
