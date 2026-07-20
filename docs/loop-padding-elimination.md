# Padding model + retiring test-loop-padding (two-layer: alignment vs compute-tile)

Status: **decided**. Earlier drafts argued for *eliminating* padding via variable-extent
DMA (rejected), then for *porting* the pass as-is (also wrong -- it over-materializes).
The settled answer (grounded in `docs/tpu_layout_padding_report.md`): padding is **two
layers** -- (A) lane/sublane alignment is materialized traffic, (B) compute-block tail is
masked compute-util -- and `test-loop-padding`'s post-codegen heuristic is replaced by
informed emission at the scheduling/codegen layer. See "RESOLVED MODEL" below for the
authoritative conclusion; the earlier sections are the analysis trail.

## DECISION: the modeled NPU has no partial-extent DMA -> padding is fundamental

We model an NPU whose DMA **always moves full tiles** (TPU-class dense movement); it
does **not** do partial / variable-extent transfers. Therefore:

- Padding (full-tile DMA over padded buffers) is a **real architectural cost**, not a
  simulator convenience. Moving the padding bytes is what the hardware actually does.
- "Handle the tail instead of padding" (variable-extent DMA, boundary clamp) would model
  a *different machine* (one with partial-transfer DMA) and would under-count traffic /
  cycles. **Rejected.**
- You cannot have "logical DRAM + full-tile boundary traffic" -- moving a full tile
  requires the padded bytes to exist in DRAM. So eliminating the padded buffer is
  incompatible with the full-tile-DMA model. The two are linked.

Consequence: **keep padding, but do NOT port the current mechanism** -- reimplement it
at the layer that has the information. Why the current C++ pass is fundamentally wrong
(not just buggy):

- It runs **after codegen** and **reverse-engineers** the padding need from the emitted
  IR: it walks `affine.for` step sizes, `dram_stride`, and `affine.apply` maps to *guess*
  which memrefs to grow, by how much, and how to rewrite addressing. The info it needs
  (tile size, tensor shape, which dims are reduction vs parallel, the access map) was
  **known at codegen time and thrown away**, then heuristically reconstructed.
- That reconstruction is inherently partial: hardcoded conv geometry (k_h/k_w/o_h/o_w),
  "find the stride", coefficient-from-dim guessing, etc. New op patterns / multi-dim /
  edge cases break it. **It cannot be shown to cover all cases** -- it is a heuristic
  retrofit, not a derivation.

Correct plan: **decide padding at the scheduling / codegen layer**, where the tile `T`
and extent `E` are *known*. When `T` does not divide `E`, the codegen knows
`E' = ceil(E/T)*T` directly and emits the padded buffer + full-tile loops/DMA **by
construction** -- no post-hoc IR analysis, no guessing. This eliminates the heuristic
pass (`test-loop-padding` -> gone, mlir-opt drops out) while **preserving padding**
(padded buffers + full-tile DMA traffic) and being robust by derivation rather than
inference.

This is "scheduling-level padding" -- but to *produce* padding correctly, NOT to
eliminate it (the earlier tail-handling framing). Padding stays; only the mechanism
moves from a fragile post-codegen heuristic to direct emission from the tiling decision.

Also resolved for free: the CI robustness bug (current pass uses `emitError` without
`signalPassFailure` -> error paths exit 0 and silently drop `@wrapper_kernel` ->
cryptic `undefined reference to wrapper_kernel` at link under autotune + unseeded
Poisson). Direct emission has no such silent-guess failure mode.

## RESOLVED MODEL: padding is TWO layers (see tpu_layout_padding_report.md)

The TPU layout/padding investigation (`docs/tpu_layout_padding_report.md`) settles the
debate: padding is **not one thing**. There are two layers with *different cost
semantics*, and conflating them was the source of the back-and-forth above.

| layer | what | on real TPU | PyTorchSim cost |
|---|---|---|---|
| **(A) lane/sublane alignment** (8x128; T(2,128)/T(4,128) for small 2nd-minor; bf16 T(8,128)(2,1)) | tile must be address-aligned -> tensor stored padded in HBM | **materialized** (`nofold`); tensor physically bigger, unavoidable | **footprint + DMA traffic** (padded bytes are stored AND moved) |
| **(B) compute-block (MXU tile) boundary** (>8x128) | the contraction/output block tail when a dim isn't a multiple of the MXU block | **masking / peeling** (mostly NOT materialized); MXU computes zeros then masks output | **compute utilization only** (wasted MXU cycles) -- **NOT traffic** |

This corrects both earlier extremes:
- "eliminate all padding" (tail-handling) -- WRONG: (A) is materialized real traffic.
- "materialize all padding" (current loop-padding) -- WRONG: it buffer-grows (B) too,
  so it **double-counts (B) as traffic**; TPU masks (B). loop-padding over-materializes.

### The two-cost-function rule (report 7.1 -- the key modeling constraint)
- **footprint / HBM traffic** function: count ONLY (A) lane/sublane alignment padding as
  physical size (e.g. extent 100 -> stored/moved as 128). Reflect bf16 packing and the
  small-2nd-minor T(2,128)/T(4,128) variants.
- **compute-utilization** function: (B) compute-block tail lowers MXU utilization via
  masking; **do not add it as traffic** (would over-estimate bandwidth). Only the rare
  alignment-forced `tensor.pad` materialization adds copy traffic.
- Pipeline ordering (report 1): the layout **decision** (which axis is lane, how much
  alignment padding) is early/metadata; **materialization** is late. Matches "decide at
  scheduling, materialize at codegen."

### Corrected plan for test-loop-padding
Reimplement at the scheduling/codegen layer (informed by `tile_desc`, which already has
the vlane axis + tile sizes), splitting the two layers:
- **(A)** materialize lane/sublane alignment padding -> the padded staging buffer +
  full-tile DMA (this is the `wrapper_kernel`-style staging; structurally necessary
  because the real DRAM tensor is logical -- you cannot pad it in place). Counts as
  footprint + traffic.
- **(B)** handle the compute-block tail by **masking** (`get_mask` already exists) ->
  compute-util only, NOT a buffer grow, NOT extra traffic.
Then the post-codegen heuristic `test-loop-padding` is gone, padding is faithful per
layer, and the modeled hardware is unchanged. (Open: whether to force (A) to fixed
(8,128)/T(packing,128) granularity -- robust, matches TPU -- vs minimal `ceil`.)

Validation (report 2.3 / 7.4): dump real XLA layouts with
`XLA_FLAGS="--xla_dump_to=... --xla_dump_hlo_as_text=true"` and read the `:T(...)`
annotations to ground the lane-axis + alignment-padding model against the compiler,
rather than guessing.

---
(Below: the earlier elimination analysis, retained as the record of WHY tail-handling
was rejected. The "fundamental vs not" framing still correctly explains the *mechanics*;
the conclusion -- that padding is eliminable -- is overturned by the DECISION above,
because on a full-tile-DMA NPU the padding traffic is a real cost that must be modeled.)

## Original goal (SUPERSEDED -- kept for the analysis trail)

`-test-loop-padding` is the only C++ MLIR pass still invoked in `mlir-opt` (after
build_tog, dma_fine_grained, lower_to_vcix, lower_dma_to_gemmini, lower_vlane_idx were
ported to Python). The original goal was to **eliminate it** by handling
tile-vs-extent misalignment at the scheduling layer. (Superseded: see DECISION -- we
port, not eliminate.)

## What test-loop-padding does today (fact, from TestLoopPadding.cpp)

Runs on the post-codegen MLIR of `@kernel`:

1. For each `affine.for`, round the upper bound **up to a multiple of its step**
   (= tile size): `paddedUpperBound = roundUpToMultiple(upperBound, stepSize)`.
2. For every DRAM `memref` indexed by a padded loop: **resize the memref** to the
   padded extent (`modifyMemrefWithPadding`), **update the func signature**
   (`updateFunctionSignatureWithMemRef`), and **rewrite `dram_stride` + the
   `affine.apply` addressing maps** so the addressing matches the larger buffer.
   Has a conv2d-specific path (nested `affine.apply` over k_h/k_w/o_h/o_w).
3. `timing_mode=1`: skip copying the padding region (cycles only, no real data).

Net: loop trip counts and the **DRAM-side buffers** are grown to aligned sizes after
codegen, with addressing rewritten to match.

## Where padding lives today (the split = the problem)

| layer | mechanism | what it pads |
|---|---|---|
| Python tile selection (`mlir_common`: `apply_divisor("pad")`, `pad_vlane_tile`, `roundup_vectorlane`) + recompile-dance (`RecompileSignal`) | pads the **tile size** to vlane/divisor multiples; forces tiles via restart | the tile shape |
| Python `get_mask` (vector tail) | masks the unaligned tail **within a tile** for vector ops | partial-tile compute |
| MLIR `test-loop-padding` | rounds **loop trip count** + grows **DRAM buffer** + rewrites strides/maps | the iteration domain + DRAM side |

Three mechanisms, three layers, one underlying concern (tile does not divide extent).

## The mismatch model

A dim has logical extent `E` and a chosen tile `T`. If `T | E`, everything is
aligned and loop-padding is a no-op. If `T does not divide E`, the last tile is
partial (`E mod T` elements). Two ways to make the hardware see full tiles:

- **Pad**: treat the extent as `E' = ceil(E/T)*T`; the loop runs `E'/T` full tiles;
  the tail tile covers padding (garbage / zero) that must not corrupt results.
- **Mask**: keep `E` and mask the partial tile so padding lanes/rows are inert.

Today: tiles are padded in Python, the within-tile tail is masked (`get_mask`), and
the loop+DRAM are padded in MLIR. We want one coherent story.

## Why padding exists -- what is fundamental vs not

Three *distinct* things happen at a tile boundary; only one is layout-fundamental, and
it is not what loop-padding does.

1. **Parallel-dim tail** (M, N, output spatial): the last tile just produces fewer
   outputs. Process fewer -- no value-fill, no padding; only "don't read/write past the
   real extent."
2. **Reduction-dim tail** (matmul K, reduce axis): the inactive elements must contribute
   the **reduction identity** (0 for sum, -inf for max) or the result is corrupted. This
   is real -- but it is satisfied either by buffer value-fill (loop-padding) OR by
   masked/identity-fill at compute/push granularity. Evidence the latter already exists:
   the matmul vcix lowering pushes `zero_vector` for the K-tail (`i >= K`), and
   `get_mask` masks vector-reduction tails. So no *buffer* padding is required for this.
3. **DMA boundary**: a full-tile transfer would run past the real tensor -> **clamp the
   transfer extent** (Q1=(c)). No buffer growth.

loop-padding bundles #2 (value-fill) + #3 (buffer grow) into one post-codegen step to
avoid per-tile tail logic. **None of that is fundamental** -- (c) replaces it with tail
handling (extent clamp + mask/push-zero). So: "handle the tail well and you don't need
separate padding" is correct, for everything loop-padding does.

### The one genuinely-fundamental padding: lane / sublane granularity

TPU pads to **(8, 128)**. These are two *different* kinds of padding:
- **128** = the MXU systolic dimension. Full tiles are a *throughput* choice, and the
  contraction tail is identity-fillable (see #2). Tail-handleable; not fundamental as
  buffer padding.
- **8** = the VMEM **sublane** -- memory is physically tiled in (8, 128) banks. This is a
  **physical layout granularity**: a dimension laid across lanes/sublanes must be a whole
  number of lane-tiles. You cannot have a "partial lane" in a lane-banked memory, and
  **masking does not help** -- the issue is the data's physical placement, not which
  compute lanes are active.

PyTorchSim mirrors this: the SRAM scratchpad is lane-banked (`vlane_idx * vu_sram_byte`;
see the MVIN layout). `pad_vlane_tile` / `roundup_vectorlane` pad the vlane-split tile dim
to a multiple of the lane count. **This padding is layout-fundamental and is RETAINED in
(c)** -- but it is *internal SRAM* padding (the spad is over-allocated per lane), cheap,
and never touches DRAM.

Conclusion: padding splits into
- (a) **layout-fundamental** lane/sublane padding (the TPU "8") -> KEEP, internal SRAM,
  cheap; and
- (b) **compute/bound** padding (reduction identity + DMA bounds; the TPU "128" + DRAM
  buffer grow) -> NOT fundamental, replaced by tail handling.

`test-loop-padding` is entirely in category (b) -> eliminable. So TPU pads to 128/8 not
because tail handling is *impossible*, but because (i) for 128 it is a dense-throughput
design choice (TPU avoids runtime masking), and (ii) for 8 it is a real lane-banked
*memory layout* constraint -- which we already satisfy with cheap internal SRAM padding,
independent of loop-padding.

## Proposed design (sketch -- to refine together)

At the scheduling / codegen-prep layer, when a dim's extent is not a multiple of its
chosen tile:

1. Compute `padded_extent = roundup(extent, tile)` at tile-selection time (the layer
   already knows the tile).
2. Emit the loop nest with **padded bounds** and size the spad/DRAM tile descriptors
   to `padded_extent` -- i.e., produce what loop-padding produces, but at emit time,
   so the addressing maps / `dram_stride` are correct by construction.
3. Reuse `get_mask` for boundary correctness (no real data movement / compute on the
   padding region).
4. `test-loop-padding` then has nothing to do -> delete it; drop `-test-loop-padding`
   from `extension_codecache`; `mlir-opt` is gone.

## Design decisions

**Q1 (crux): DRAM buffer resize vs the real tensor. RESOLVED -> (c) hybrid.**
loop-padding grows the DRAM function-arg `memref`; the real `npu` tensor is
logical-sized. Decision: **pad the SRAM (spad) tile fully** (cheap, internal -- the
spad is already over-allocated per-lane), **keep DRAM logical**, and **clamp the
boundary tile's DMA** to the real tail so no OOB DRAM access happens. No device
buffer growth, no func-signature/stride rewrite -> genuinely scheduling-level.
(Rejected: (a) over-allocating the device DRAM buffer -- leaks into PyTorchSimDevice
allocation and needs a logical/padded shape bridge; (b) was the same as (c) minus the
explicit SRAM-full-padding framing.)

Consequence: the loop still iterates `ceil(E/T)` tiles (padded trip count) and the
spad tile is full `T`, but for the **last tile** the DMA moves only `E - (ceil(E/T)-1)*T`
real rows; the spad tail rows are garbage, kept inert in compute by `get_mask`.

### PRECONDITION: index/data-dependent tail handling

(c) means the **last (tail) tile must behave differently** from the rest: move only
`E - (ceil(E/T)-1)*T` real rows from logical DRAM, leave the spad tail inert. That is
an **index-dependent operation** -- behavior varies with the loop induction variable.

- **Compute side already supports this.** `get_mask` builds a per-iteration predicate
  `step_vec < (upper_bound - compute_idx)` (depends on the loop index), masking the
  tail lanes. Index-dependent masking already exists for vector compute.
- **DMA side does NOT.** The DMA transfer length today = the spad tile shape, a
  **compile-time constant**. loop-padding exists precisely to avoid a variable-length
  DMA: it grows DRAM so even the boundary reads a full `T`. Remove loop-padding and the
  boundary DMA must transfer a **variable (index-dependent) length** -- a capability the
  customized `memref.dma_start` / Spike MVIN does not have today.

**So the gating precondition is: the DMA must support an index-dependent transfer
extent** (move `min(T, E - i*T)` rows for tile `i`). Establishing that is the real
foundation of this work; without it, scheduling-level (c) padding cannot be expressed.

### Q1a: how to satisfy the precondition

  - **Variable-extent DMA (data-dependent).** Extend the customized `memref.dma_start`
    + lowering + Spike MVIN to accept a runtime transfer length, and emit
    `affine.min(T, E - i*T)` per tile. General (scales to multi-dim, any extent),
    uniform loop body. Cost: a real hardware-model + descriptor extension. This is the
    capability the precondition names.
  - **Static tail-peel.** `floor(E/T)` full-tile iterations + a separate compile-time
    partial DMA. No variable-length DMA needed, but **combinatorial in the number of
    unaligned dims** (2^k corner DMAs) -- the same blow-up that made the decompose
    unroll-peel a dead end (#258). Does not scale; rejected as the general path.
  Leaning: the variable-extent DMA is the principled answer -- it is the missing
  capability, and it generalizes.

### Finding: variable extent is a codegen change, not a Spike change

The transfer dim sizes are packed into the CONFIG instruction's **rs1 register**
(`lower_dma_to_gemmini.py:144`): today `cfg_rs1 = i64_const((shape4[0]&0xFFFF)<<48 |
... )` -- a compile-time constant. But `asm(CONFIG, cfg_rs1, cfg_rs2)` passes rs1 as a
**register operand**, and Spike reads the dim sizes from it into `P.VU.dma_dim_size`.
So the hardware model already takes the extent at runtime; today's codegen merely feeds
a constant.

Implication: the variable-extent precondition is satisfiable **at the codegen /
lower_dma_to_gemmini layer, with no Spike change** -- emit `cfg_rs1` as a *computed*
value (pack a runtime dim size `min(T, E - i*T)` into the 16-bit field via arith)
instead of `i64_const`. The boundary tile then moves only the real tail; the spad stays
fully padded; DRAM stays logical. This is exactly (c).

Evidence (strong): the CONFIG instruction is `CUSTOM_1` funct7=0 with rs1/rs2 as
**register operands** (`asm(func7, rs1, rs2)` -> `.insn r CUSTOM_1, 0x3, func7, x0,
$0, $1`); the MVIN reads the dims from `P.VU.dma_dim_size`, which is runtime VU state
set by that CONFIG insn. A config insn reading rs1 register bits into dma_dim_size is
the only sensible implementation -- so runtime-variable extent is already supported by
the model; today's codegen just feeds a constant rs1. (One file unread: the exact
torchsim config insn in riscv-isa-sim -- verify it stores all four 16-bit dim fields
from rs1.)

Remaining work to thread it: (1) carry a dynamic per-axis transfer extent on the
customized `memref.dma_start` (a length operand, like the existing dynamic indices);
(2) `lower_dma_to_gemmini` builds `cfg_rs1` from those (constant when static, arith when
dynamic); (3) the scheduling layer computes `affine.min(T, E - i*T)` for unaligned dims
and passes it.

**Q2: where is the transfer shape threaded? RESOLVED -> pass shape as an operand,
computed separately.** Compute the real (boundary-clamped) per-axis transfer extent in
a separate step (the scheduling layer, via `affine.min(T, E - i*T)` / arith) and pass
it to the customized `memref.dma_start` as an explicit **shape operand** (alongside the
existing dynamic index operands). `lower_dma_to_gemmini` then packs `cfg_rs1` from that
operand (constant-folds when static). Keeps the extent an explicit, separately-computed
value rather than implicit in the descriptor's static memref shape.

**Q3: timing-mode skip-copy equivalent.** Padding iterations must cost cycles but move
no real data: loop = padded, DMA size = real-tail (the Q2 shape operand). Confirm
`get_mask` covers the compute side and the shape operand covers the DMA side.

**Q4: conv2d -- NOT a special case under (c).** loop-padding has a conv-specific branch
(it walks the nested `affine.apply` over k_h/k_w/o_h/o_w and rewrites the maps +
`dram_stride`). That complexity exists because loop-padding **grows the DRAM buffer and
rewrites addressing** -- and conv's input address is a nested affine
(`input_row = o_h*stride + k_h - pad`), so growing the buffer forces rewriting those
nested maps.

Under (c) we do neither: we keep DRAM logical and only **clamp the boundary tile's DMA
extent**, leaving the address computation untouched. A DMA is a rectangular DRAM<->SRAM
block transfer (base address + per-axis extents); "don't read past the DRAM end" is a
per-axis `min(tile, remaining)` regardless of gemm vs conv. The conv composite indexing
affects *where* the block starts (address) and *how* compute consumes it (handled by
`get_mask`), not *how many* elements move (the per-axis extent clamp). So conv reduces to
the same flat per-axis clamp as gemm; loop-padding's conv branch has no counterpart here.
(Residual check, not a design fork: conv tiles overlap (halo from stride/kernel) so input
tiles are not a clean DRAM partition -- confirm the per-axis "remaining" is still just
`E - base_on_that_axis`, which it is, since each tile's address is independent.)

**Q5: incremental path.** Start by deleting loop-padding for kernels where no dim
needs padding (it is already a no-op there) and confirm zero diff; then handle the
real padding cases, possibly behind a flag, validating each.

## Validation

e2e suite (gemm/bmm/conv2d with deliberately non-multiple extents, + models),
`allclose` through Gem5+Spike+TOGSim. Where feasible, structurally compare the emitted
MLIR against the current `-test-loop-padding` output on the same kernels.

## Relation to prior work

Same move as floor/mod -> axis-split + graph-copy: handle the misalignment upstream so
the MLIR layer sees only the clean (aligned) case and the pass disappears. This is the
padding instance of that principle (Plan A in `dma-transfer-lowering.md`).
