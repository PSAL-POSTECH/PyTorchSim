"""Unit tests for the lower_transfer_to_gemmini pass (no torch, no Spike).

Hand-write a `togsim.transfer`, run the pass in-process, and assert the lowered IR
matches an embedded golden -- the exact expected output (MLIR SSA numbering is
deterministic for a fixed input). If a lowering change is intended, regenerate the
golden and review the diff. Skipped without the MLIR bindings.
"""
import importlib.util
import textwrap

import pytest

_MLIR = importlib.util.find_spec("mlir") is not None
pytestmark = pytest.mark.skipif(not _MLIR, reason="MLIR Python bindings not installed")


def _lower_transfer(ir_text, timing=False):
    from mlir.ir import Context, Module, Location
    from PyTorchSimFrontend.mlir.passes import lower_transfer_to_gemmini as L
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(ir_text)
        L.run(m, timing=timing)
        return str(m).strip()


# INPUT: one masked MVOUT of a [1,2,8,8] tile, with positional togsim.transfer operands
# (see emit_transfer). masked_axes=[2] is a SPARSE overlay clamping tile axis 2 to
# [0, 7), so the ragged tail is skipped on store; unlisted axes default to no clamp.
_MASKED_MVOUT = """
module {
  memref.global @buf1_spad : memref<1x2x8x8xf32, 1>
  func.func @k(%arg1: memref<2048xf32>) {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<1xi32>
    %s = memref.get_global @buf1_spad : memref<1x2x8x8xf32, 1>
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    "togsim.transfer"(%arg1, %c0, %s, %c0, %alloc, %c0, %c3, %c1, %c0, %c7) {dma_kind="MVOUT", dram_stride=[128,64,8,1], tile_stride=[128,64,8,1], vlane_split_axis=3:i64, masked_axes=[2], masked_fill=0:i64} : (memref<2048xf32>, index, memref<1x2x8x8xf32,1>, index, memref<1xi32>, index, index, index, index, index) -> ()
    return
  }
}"""

# OUTPUT (golden): a 36-i32 @dma_desc_0 (dim_size, dim_low, dim_high, dram/tile strides,
# a packed flags slot), the masked clamp written into it at RUN TIME, then two
# `.insn r CUSTOM_1` ops: func7=7 CONFIG_DESC hands over the descriptor, func7=3 MVOUT.
_MASKED_MVOUT_LOWERED = textwrap.dedent("""\
    module {
      memref.global "private" @dma_desc_0 : memref<36xi32> = dense<[1, 2, 8, 8, 0, 0, 0, 0, 1, 2, 8, 8, 128, 0, 64, 0, 8, 0, 1, 0, 128, 0, 64, 0, 8, 0, 1, 0, 65540, 131843, 0, 0, 0, 0, 0, 0]>
      memref.global @buf1_spad : memref<1x2x8x8xf32, 1>
      func.func @k(%arg0: memref<2048xf32>) {
        %c0 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %alloc = memref.alloc() : memref<1xi32>
        %0 = memref.get_global @buf1_spad : memref<1x2x8x8xf32, 1>
        %c3 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %c0_0 = arith.constant 0 : index
        %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<2048xf32> -> index
        %c4 = arith.constant 4 : index
        %1 = arith.muli %c0, %c4 : index
        %2 = arith.addi %intptr, %1 : index
        %3 = arith.index_cast %2 : index to i64
        %intptr_1 = memref.extract_aligned_pointer_as_index %0 : memref<1x2x8x8xf32, 1> -> index
        %c128 = arith.constant 128 : index
        %4 = arith.muli %c0_0, %c128 : index
        %c64 = arith.constant 64 : index
        %5 = arith.muli %c0_0, %c64 : index
        %6 = arith.addi %4, %5 : index
        %c8 = arith.constant 8 : index
        %7 = arith.muli %c0_0, %c8 : index
        %8 = arith.addi %6, %7 : index
        %9 = arith.addi %8, %c0 : index
        %c4_2 = arith.constant 4 : index
        %10 = arith.muli %9, %c4_2 : index
        %11 = arith.addi %intptr_1, %10 : index
        %12 = arith.index_cast %11 : index to i64
        %13 = memref.get_global @dma_desc_0 : memref<36xi32>
        %14 = arith.index_cast %c0 : index to i32
        %c6 = arith.constant 6 : index
        memref.store %14, %13[%c6] : memref<36xi32>
        %15 = arith.index_cast %c7 : index to i32
        %c10 = arith.constant 10 : index
        memref.store %15, %13[%c10] : memref<36xi32>
        %intptr_3 = memref.extract_aligned_pointer_as_index %13 : memref<36xi32> -> index
        %16 = arith.index_cast %intptr_3 : index to i64
        %c0_i64 = arith.constant 0 : i64
        llvm.inline_asm has_side_effects asm_dialect = att ".insn r CUSTOM_1, 0x3, 7, x0, $0, $1", "r,r,~{dirflag},~{fpsr},~{flags}" %16, %c0_i64 : (i64, i64) -> ()
        llvm.inline_asm has_side_effects asm_dialect = att ".insn r CUSTOM_1, 0x3, 3, x0, $0, $1", "r,r,~{dirflag},~{fpsr},~{flags}" %3, %12 : (i64, i64) -> ()
        return
      }
    }""")


def test_masked_transfer_lowering_golden():
    assert _lower_transfer(_MASKED_MVOUT) == _MASKED_MVOUT_LOWERED


# Same transfer with subtile_size=[1,1,4,8] (the H axis is subtiled 8 -> 4). The
# descriptor's dim_size is the SUBTILE (config) shape, not the full tile; the masked
# axis + clamp stores are unchanged (a 4D subtile -> expand 0 -> same idx 6/10).
_MASKED_MVOUT_SUBTILE = _MASKED_MVOUT.replace(
    'vlane_split_axis=3:i64,', 'vlane_split_axis=3:i64, subtile_size=[1,1,4,8],')

_MASKED_MVOUT_SUBTILE_LOWERED = _MASKED_MVOUT_LOWERED.replace(
    "dense<[1, 2, 8, 8, 0, 0, 0, 0, 1, 2, 8, 8,",
    "dense<[1, 1, 4, 8, 0, 0, 0, 0, 1, 1, 4, 8,")


def test_masked_transfer_subtile_lowering_golden():
    assert _lower_transfer(_MASKED_MVOUT_SUBTILE) == _MASKED_MVOUT_SUBTILE_LOWERED


def test_timing_mode_erases_transfer():
    # In timing mode the TOG carries DMA timing -> the transfer + descriptor are erased.
    out = _lower_transfer(_MASKED_MVOUT, timing=True)
    assert "togsim.transfer" not in out and "dma_desc" not in out
