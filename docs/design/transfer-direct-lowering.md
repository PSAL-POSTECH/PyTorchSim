# Direct togsim.transfer -> Gemmini lowering (drop memref.dma_start)

## Why
`togsim.transfer` currently lowers `decompose_transfer` -> `memref.dma_start` ->
... -> `lower_dma_to_gemmini` -> gemmini asm. `memref.dma_start` is a REGISTERED
MLIR op with a fixed operand list, so it cannot carry extra RUNTIME operands. The
indirect offset worked around this with a static symbol attribute; the masked-DMA
`low`/`high` clamp bounds are RUNTIME index values (dynamic shapes) and cannot be
attributes -- so they cannot ride on `memref.dma_start` at all.

Fix: keep `togsim.transfer` (unregistered -> arbitrary operands) alive through the
pipeline and lower it DIRECTLY to gemmini, carrying every runtime descriptor
(dram_idx, vlane, offset, low/high) as operands. This also lets `test-loop-padding`
retire later (the clamp replaces it), collapsing the mlir-opt step.

## Target transfer op (superset, all optional beyond the base)
```
togsim.transfer(dram, dram_idx, sram, sram_idx, tag, dma_type, vst,
                [offset_spad],            # indirect (gather/scatter)
                [low_vec, high_vec])      # masked clamp: per-dim valid [low, high)
  attrs: dma_kind, dram_stride[], tile_stride[], padding, [subtile_size, async]
```
- no `low`/`high`  -> full transfer `[0, memref_dim)` (current behaviour).
- no `offset`      -> base address (current behaviour).
- `low`/`high` are runtime index operands (vectors length ndim), NOT attributes.

## Semantics of the clamp (step 2; recorded here for the target)
dest tile stays FULL (banking unchanged -> no vlane shift); only the SOURCE sub-box
`[low, high)` per dim is read; dest positions outside it get `fill` (= ops.masked
`other`). This is "source subview + full dest + fill", not a dest subview.

## memref.dma_start consumers to port (the scope -- all-or-nothing)
1. `lower_dma_to_gemmini` (dma_start -> asm): consume togsim.transfer; ABSORB
   decompose_transfer's 4D handling (<=4D direct / collapse unit dims / >4D
   affine.for peel + subview + lane-banked phys offset).
2. `dma_fine_grained` (matmul subtile split, marker=subtile_size): split
   togsim.transfer instead of dma_start (same loop structure/offsets).
3. `lower_to_vcix._DmaView` (positional read for compute coord): read
   togsim.transfer operand layout.
4. `test-loop-padding` (C++ mlir-opt): verify it tolerates an unregistered
   togsim.transfer in the module (allow-unregistered) and pads loops as before.

## Staging (step 1 = behaviour-preserving; regression 0; low/high NOT added yet)
- 1i. Draft merged `lower_transfer_to_gemmini` (transfer -> asm + 4D handling) in a
      NEW file. A/B validate its gemmini asm against the current
      decompose->...->lower chain on sample post-vcix MLIR (gemm, conv, pad,
      pointwise, gather). NOT wired into the live pipeline yet.
- 1ii. Port `dma_fine_grained` + `_DmaView` to the togsim.transfer operand layout.
- 1iii. Remove `decompose_transfer` from PRE_OPT; rewire so togsim.transfer survives
       to the merged lowering. Confirm test-loop-padding still runs (unregistered).
- 1iv. Full regression (the CI allowlist locally): add/matmul/conv/attention/pad/
       gather/models. Gate = regression 0.

## Step 2 (feature, on top of step 1)
- Add `low`/`high` operands to `emit_transfer` (from load index-affine intersect
  operand shape, D2) + `fill`; merged lowering emits the clamp into the gemmini
  CONFIG/MVIN; Spike MVIN gates per-position `low[d] <= idx[d] < high[d]` else fill.
  Drop the ops.masked `where` for pure-load bodies; keep it for compute-mixed.
- Gate: constant_pad correct with divisibility ON (regression 0), then P5 removes
  divisibility -> mobilenet wrapper3 holes/segfault fixed.

## Validation harness
A/B equivalence: for each stage, run the OLD chain and the NEW pass on the same
post-vcix MLIR dump and diff the emitted gemmini asm (structurally: same
CONFIG/CONFIG2/3/[4]/MVIN sequence, same packed rs1/rs2, same loop nests). Then the
end-to-end Spike allclose + timing tests.
