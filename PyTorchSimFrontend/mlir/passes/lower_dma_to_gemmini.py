"""Lower customized memref.dma_start ops to Gemmini RISC-V inline asm.

Python port of the C++ test-memref-to-gemmini conversion. Each memref.dma_start
(carrying dram_stride / sram_stride / subtile_size attrs and vlane params encoded
in its stride / num_elements_per_stride / num_elements operands) becomes a
sequence of `llvm.inline_asm` ".insn r CUSTOM_1 ..." Gemmini instructions:
config_mvin/mvout, config2 (dram strides), config3 (spad strides), then the
mvin/mvout itself with the DRAM and scratchpad byte addresses.

The conversion-framework coupling of the C++ pass (LLVMTypeConverter,
getStridedElementPtr, MemRefDescriptor) is avoided by working at the memref level:
addresses are computed with `memref.extract_aligned_pointer_as_index` + arith,
and the existing standard MLIR->LLVM lowering finalizes everything. Pass order:
this runs on memref-level IR (after test-pytorchsim-to-vcix), before
run_standard_lowering.

NOTE: indirect-access (gather) dma_start is not yet handled (Phase 2); such ops
raise so they are caught rather than silently mishandled.
"""

OP_NAME = "memref.dma_start"
WAIT_NAME = "memref.dma_wait"
MARKERS = (OP_NAME, WAIT_NAME)

from ._mlir_util import attr_i64_array

# func7 instruction codes (CustomDMAAttribute.h)
CONFIG, CONFIG2, CONFIG3, CONFIG4 = 0, 4, 5, 6
MVIN, MVIN2, MVIN3, MVOUT = 2, 1, 14, 3
CONFIG_TYPE = {MVIN: 0, MVIN2: 1, MVIN3: 2, MVOUT: 3}
MAX_TENSOR_DIM = 4
CONSTRAINTS = "r,r,~{dirflag},~{fpsr},~{flags}"


def _asm(func7):
    return f".insn r CUSTOM_1, 0x3, {func7}, x0, $0, $1"


def _i64_signed(v):
    """Wrap an unsigned 64-bit packed value into signed int64 (matches C++ getI64IntegerAttr)."""
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def run(module, timing=False):
    """Lower memref.dma_start / dma_wait to Gemmini instructions.

    timing=False (functional/Spike): dma_start -> gemmini config + mvin/mvout asm.
    timing=True  (gem5 cycle path):  dma_start is erased (the TOG already carries
                                     DMA timing; the cycle binary needs no asm).
    memref.dma_wait is erased in both modes (matches C++ DmaWaitOpLowering).
    """
    from mlir.ir import (InsertionPoint, Operation, IntegerType, IndexType,
                         IntegerAttr, MemRefType, FlatSymbolRefAttr, TypeAttr)
    from mlir.dialects import llvm, arith, memref

    i64 = IntegerType.get_signless(64)
    idx = IndexType.get()

    # memref.global symbol -> type, to resolve the indirect_offset spad
    sym2type = {}
    for g in module.operation.regions[0].blocks[0].operations:
        if g.operation.name == "memref.global":
            sym2type[g.attributes["sym_name"].value] = MemRefType(TypeAttr(g.attributes["type"]).value)

    def const_int(val):
        return IntegerAttr(val.owner.attributes["value"]).value

    def i64_const(value):
        return arith.ConstantOp(i64, IntegerAttr.get(i64, _i64_signed(value))).result

    def asm(func7, rs1, rs2):
        llvm.InlineAsmOp(None, [rs1, rs2], _asm(func7), CONSTRAINTS,
                         has_side_effects=True, asm_dialect=0)

    def elem_addr_i64(memref_val, indices, mtype, elem_bytes):
        """i64 byte address of memref_val[indices] (aligned ptr + linear elem offset)."""
        base = memref.ExtractAlignedPointerAsIndexOp(memref_val).result  # index = byte addr
        strides = _row_major_strides(list(mtype.shape))
        off = None  # element offset (index)
        for k, ival in enumerate(indices):
            if strides[k] == 0:
                continue
            term = ival
            if strides[k] != 1:
                term = arith.MulIOp(ival, arith.ConstantOp(idx, IntegerAttr.get(idx, strides[k])).result).result
            off = term if off is None else arith.AddIOp(off, term).result
        if off is not None:
            byte = arith.MulIOp(off, arith.ConstantOp(idx, IntegerAttr.get(idx, elem_bytes)).result).result
            base = arith.AddIOp(base, byte).result
        return arith.IndexCastOp(i64, base).result

    starts, waits = [], []
    for region in module.operation.regions:
        for b in region.blocks:
            _collect(b, starts, waits)

    for op in waits:        # dma_wait: erase in both modes
        op.erase()

    for op in starts:
        if timing:          # gem5 cycle path: drop the dma_start (TOG has timing)
            op.erase()
            continue
        operands = list(op.operands)
        src, dst = operands[0], None
        src_ty = MemRefType(src.type)
        src_rank = len(src_ty.shape)
        dst = operands[1 + src_rank]
        dst_ty = MemRefType(dst.type)
        dst_rank = len(dst_ty.shape)
        src_idx = operands[1:1 + src_rank]
        dst_idx = operands[1 + src_rank + 1:1 + src_rank + 1 + dst_rank]

        dma_type = const_int(operands[1 + src_rank + 1 + dst_rank])  # num_elements
        vlane_split_axis = const_int(operands[-2])       # stride (always 2nd-to-last)
        vlane_stride = const_int(operands[-1]) & 0x7FFF  # num_elements_per_stride (last)
        is_mvin = dma_type in (MVIN, MVIN2, MVIN3)

        elem_bytes = _elem_bytes(src_ty.element_type)
        # Indirect (gather): offset spad referenced by the indirect_offset symbol attr
        indirect = "indirect_offset" in op.attributes

        tile_shape = _subtile(op)
        if tile_shape is None:
            tile_shape = list(dst_ty.shape) if is_mvin else list(src_ty.shape)
        dram_strides = attr_i64_array(op, "dram_stride")
        spad_strides = attr_i64_array(op, "sram_stride")
        assert len(tile_shape) == len(dram_strides) == len(spad_strides), \
            f"shape/stride rank mismatch: {tile_shape} {dram_strides} {spad_strides}"

        expand = MAX_TENSOR_DIM - len(tile_shape)
        shape4 = [1] * expand + tile_shape
        dram4 = [0] * expand + dram_strides
        spad4 = [0] * expand + spad_strides
        vlane_split_axis += expand
        config_type = CONFIG_TYPE[dma_type]

        with InsertionPoint(op):
            addrA = elem_addr_i64(src, src_idx, src_ty, elem_bytes)
            addrB = elem_addr_i64(dst, dst_idx, dst_ty, elem_bytes)
            dram_addr, spad_addr = (addrA, addrB) if is_mvin else (addrB, addrA)

            cfg_rs1 = i64_const(((shape4[0] & 0xFFFF) << 48) | ((shape4[1] & 0xFFFF) << 32)
                                | ((shape4[2] & 0xFFFF) << 16) | (shape4[3] & 0xFFFF))
            cfg_rs2 = i64_const((vlane_stride << 32) | ((config_type & 0x3) << 17)
                                | ((1 if indirect else 0) << 16)
                                | ((vlane_split_axis & 0x3) << 14) | elem_bytes)
            asm(CONFIG, cfg_rs1, cfg_rs2)
            asm(CONFIG2, i64_const((dram4[0] << 32) | (dram4[1] & 0xFFFFFFFF)),
                i64_const((dram4[2] << 32) | (dram4[3] & 0xFFFFFFFF)))
            asm(CONFIG3, i64_const((spad4[0] << 32) | (spad4[1] & 0xFFFFFFFF)),
                i64_const((spad4[2] << 32) | (spad4[3] & 0xFFFFFFFF)))
            if indirect:
                # CONFIG4: rs1 = indirect index-spad base address, rs2 = (elem_size<<16)|stride(1)
                offset_sym = FlatSymbolRefAttr(op.attributes["indirect_offset"]).value
                off_ty = sym2type[offset_sym]
                indirect_memref = memref.GetGlobalOp(off_ty, offset_sym).result
                ind_base = memref.ExtractAlignedPointerAsIndexOp(indirect_memref).result
                ind_addr = arith.IndexCastOp(i64, ind_base).result
                ind_esize = _elem_bytes(off_ty.element_type)
                asm(CONFIG4, ind_addr, i64_const(((ind_esize & 0xFF) << 16) | (1 & 0xFFFF)))
            asm(dma_type, dram_addr, spad_addr)
        op.erase()


def _collect(block, starts, waits):
    for op in list(block.operations):
        name = op.operation.name
        if name == OP_NAME:
            starts.append(op.operation)
        elif name == WAIT_NAME:
            waits.append(op.operation)
        for region in op.operation.regions:
            for b in region.blocks:
                _collect(b, starts, waits)


def _subtile(op):
    from mlir.ir import ArrayAttr, IntegerAttr
    if "subtile_size" not in op.attributes:
        return None
    return [IntegerAttr(a).value for a in ArrayAttr(op.attributes["subtile_size"])]


def _elem_bytes(elem_type):
    from mlir.ir import IntegerType, FloatType
    bits = (IntegerType(elem_type).width if IntegerType.isinstance(elem_type)
            else FloatType(elem_type).width)
    return max(bits, 8) // 8


def lower_text(text):
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
    (open(sys.argv[2], "w").write(out) if len(sys.argv) > 2 else sys.stdout.write(out))
