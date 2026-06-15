# DMA transfer op + decomposition lowering

Status: **design / proposed**. Captures the plan to fix the recompile-dance
fragility by representing DMA as a high-level declarative transfer op and
decomposing it into affine descriptors in a lowering pass.

Companion docs: `linalg-codegen-migration.md` (Plan B, the full structured-ops
rewrite this is a narrow tactical slice of). The near-term graph-level padding
work is referred to here as Plan A.

## TL;DR

The MLIR codegen forces tile sizes so that non-affine index expressions
(`FloorDiv` / `ModularIndexing`, produced by view/reshape/cat) collapse into the
DMA's strictly-affine 4D integer-stride address model. That forcing is a lazy,
greedy, monotonic, restart-based search (the "recompile dance") capped at 5
retries; when operand constraints conflict or exceed 4D it hard-fails and the
model does not compile. This blocks model coverage, which is the primary goal.

Proposed fix: stop forcing one affine descriptor. Introduce a **high-level
`togsim.transfer` op** that carries an iteration domain plus `iter->src` /
`iter->dst` affine maps (which may legally contain floordiv/mod), and a
**decomposition pass** that lowers it to a loop of the **existing customized
`memref.dma_start` descriptors** (kept unchanged as the leaf). The non-affine /
high-rank part is peeled into a base-pointer loop instead of being crammed into
one descriptor. This inverts the tile<->DMA dependency (the DMA adapts to the
tile, not the reverse), removes the rank cap, and removes the recompile dance.

## Problem

### Root cause: an impedance mismatch

The DMA address model is `base + sum_i stride_i * idx_i`, with **integer strides,
4D**, i.e. strictly affine/linear. Inductor's index expressions are not: views,
reshapes, `cat`, and broadcasts introduce `FloorDiv` / `ModularIndexing`, which
are non-affine. The codegen copes by searching for a tiling under which the
floor/mod collapses to a linear stride within a tile -- that is exactly what the
ModularIndexing tile constraints ("tile must be a multiple of the floordiv
divisor and a divisor of the modular divisor") encode.

### The recompile dance (where it breaks)

`codegen_nodes` (`mlir_common.py`) is a `while True` loop, `max_retry_compile = 5`.
During emission, `get_dma_info` (`mlir_codegen_backend.py`) inspects the index:

- the **split path** (good): `apply_divisor(axis, divisor, "split")` peels an axis
  into two affine dims to represent floor/mod, inserting a `0` into `dram_stride`;
- the **pad path** (fragile): when the tile is not divisible it mutates the tile
  (`set_tile_size`, `tile_constraint.fixed = True`) and raises `RecompileSignal`
  to restart emission.

It breaks because:

1. **One global tile must satisfy every operand's divisibility** on a shared axis.
   Fused ops with conflicting constraints (common with reshape/modular indexing)
   cannot be satisfied at once -- this is the loop<->tensor mismatch.
2. **Greedy + monotonic + no backtracking + 5-retry cap.** `tile_constraint.fixed`
   persists across retries (the tile descriptor lives on `kernel_group`, survives
   `reset`), so the search ratchets one way; conflicting fixes oscillate and hit
   the cap -> `RuntimeError("Failed to compile kernel after multiple attempts")`.
3. **4D rank cap.** A reshape needing more than 4 affine dims after splitting
   raises `NotImplementedError`.
4. **vlane / LMUL entanglement.** Pad-forcing moves `vlane_split_axis` / relaxes
   `vlane_stride`, and `compute_vec` must be a power of two; these can be mutually
   unsatisfiable with the divisibility constraints.

Padding logic is currently spread across three places: the Python recompile/tile
-adjust dance (1), the Python `get_mask` vector-tail handling (2), and the MLIR
`TestLoopPadding` pass (3). Removing (3) alone does not fix the fragility; (1) is
the larger source.

### We already have a de-facto custom DMA op

`get_dma_code` emits `memref.dma_start` overloaded via string formatting with
extra operands (`dma_type` MVIN/MVOUT, tag, `vlane_split_axis`, `vlane_stride`)
and extra attributes (`dram_stride`, `tile_stride`, padding type). This is a
custom descriptor in all but name -- and it is what Spike / gem5 / TOGSim already
consume.

## Proposed design

Two op levels, with a pass bridging them.

```
[high]  togsim.transfer        iteration domain + iter->src / iter->dst affine maps
                               (maps MAY contain floordiv/mod; rank unbounded)
            |  decompose-transfer pass (cost-aware peel)
            v
[low]   scf.for { customized memref.dma_start }     <- existing leaf, UNCHANGED
```

### Low-level descriptor (keep as-is)

The existing customized `memref.dma_start` is the lowering target / leaf: affine,
4D, integer stride, simulator-understood. **Do not add maps or floor/mod to it** --
that would re-create the representational limit and blur the boundary. Optionally
formalize it into a real op (`togsim.dma_descriptor`) with a verifier so the pass
rewrites real ops instead of strings; not required to start.

### High-level transfer op (new)

Strawman:

```mlir
togsim.transfer
    ins(%src : memref<?x?x?xf16, #dram>)        // DRAM
    outs(%dst : memref<...xf16, 1>)             // scratchpad
    iter_bounds = [%M, %N, %K]                  // iteration domain (dynamic via SSA operands)
  attributes {
    src_map = affine_map<(m,n,k)[s0] -> (m, (n floordiv s0), (n mod s0), k)>,  // non-affine lives here
    dst_map = affine_map<(m,n,k) -> (m, n, k)>,
    vlane_split_axis = 1, vlane_stride = 4,
    dma_kind = "MVIN", tag_policy = "async",
    peel_plan = [0]      // optional: which iter dims to peel (decided in Python; see below)
  }
```

Design choices:

- **Iteration domain + two maps, not a single src->dst map.** A direct src->dst
  relation only works for bijections; broadcast (`cat([a, a])`) and non-bijective
  access need the loop-mediated form. This is the `linalg.generic` model, and peel
  becomes "tile the iteration domain."
- **floor/mod ride in the `AffineMap`.** MLIR `AffineMap` supports constant-divisor
  `floordiv`/`mod`/`ceildiv` natively; the codegen already produces these as
  strings in `convert_index`. Symbolic divisors are semi-affine -- representable,
  handled by our pass.
- **`memref`, not raw pointers.** The memref carries base + shape + layout so the
  pass can reason about strides; `src_ptr`/`dst_ptr` are inside it. Buffer shape is
  the memref type; the transfer region is `iter_bounds` + maps.
- **Closest existing op is `linalg.generic` / `linalg.copy`** (same shape + maps +
  body structure) but it lacks DMA/vlane/tag/scratchpad semantics and its tiling
  lowers to subview+scf, not our descriptors -- so a custom op modeled on linalg's
  design, reusing AffineMap utilities.

### Decomposition pass (contract)

The DMA descriptor is an **affine map of rank <= 4 with integer strides**
(`base + sum_i stride_i * idx_i`). Decide by **rank after linearization**, NOT by
the presence of floordiv/mod:

1. **Linearize** `src_map`: rewrite each `floordiv c` / `mod c` on an iteration dim
   into a split pair (`idx = outer*c + inner`), which is purely linear in the new
   dims. (This is exactly what `apply_divisor("split")` already does.) Let `D` be
   the resulting affine rank.
2. **`D <= 4`** -> emit **one** customized `memref.dma_start`; the split dims become
   the descriptor's <=4D shape/strides. Identical to today's output (fast path).
   floordiv/mod that still fits in <=4D after splitting stays here -- it is *not* a
   peel trigger.
3. **`D > 4`** (not expressible as a single linear combination) -> express it as a
   **combination of linear combinations**: peel `D - 4` dims into an outer
   `affine.for`; each iteration computes a base with `affine.apply` (the peeled
   dims' linear, incl. split-derived, contribution) and issues the inner <=4D
   affine descriptor. SRAM offsets are computed symmetrically in the same loop.
4. If the estimated descriptor count is pathological -> fall back to **relayout**.

Genuinely non-affine access (data-dependent / indirect / gather -- an index that
comes from a loaded value and cannot be linearized by splitting) is **out of scope**
for this pass; it stays on the indirect-indexing path (or a relayout).

The decision point maps onto existing code: codegen already splits floordiv/mod via
`apply_divisor` and raises `NotImplementedError` at >4D (`get_dma_info`). That exact
site becomes "emit `togsim.transfer`" instead of dying, and the recompile/tile
-forcing dance is unnecessary because the outer peel loop's `ceil` bound absorbs
non-divisible remainders.

### Relationship to memref-to-gemmini (ISA lowering) -- keep separate

`memref.dma_start` is the boundary, not the endpoint. The layering is:

    togsim.transfer  --[Python decompose]-->  memref.dma_start  --[C++ memref-to-gemmini]-->  Gemmini ISA

decompose-transfer stops at `memref.dma_start` and must **not** emit Gemmini
instructions directly. ISA lowering stays in the C++ `test-memref-to-gemmini` pass.
Rationale:

- **Separation of concerns**: decompose does descriptor decomposition (affine
  algebra: rank / peel); gemmini does instruction encoding (hardware). Different
  axes; merging couples affine logic with ISA detail.
- **`memref.dma_start` is a shared contract** with multiple consumers
  (memref-to-gemmini, dma-fine-grained, the TOG pass). Keeping it as the interface
  lets all of them stay unchanged.
- **gemmini is a conversion-framework, target-specific, stable lowering** -> it
  belongs in C++; porting it to Python would be painful and pointless. decompose is
  under design churn -> Python (fast iteration). Right tool per churn.

One constraint flows the other way: gemmini's ISA limits (max dims / size per MVIN)
set decompose's target inner-descriptor shape (the "<=4D" and max-extent bounds).
decompose must *respect* those limits when it picks what stays inner vs gets peeled
-- but respecting a constraint is not doing the lowering.

### Cost-aware peeling (this is a cycle-accurate simulator)

Descriptor count is a **modeled cost** (issue overhead + DRAM burst efficiency in
Ramulator). Rules:

1. Peel the **outermost, lowest-trip-count** dims (descriptor count = product of
   peeled extents).
2. Keep the inner descriptor **as large and contiguous as possible** (maximize
   bytes per descriptor).
3. If even the best peel is pathological, fall back to **relayout**.

### Placement: hybrid (least burden)

Keep the decision in Python (where shape/sympy info is available and iteration is
fast); keep the C++ pass purely mechanical.

| Step | Where |
|---|---|
| peel-plan decision (which dims, count estimate, peel vs relayout) | Python |
| encode plan as op attributes | Python -> MLIR |
| emit `scf.for { customized dma_start }` per the plan | C++ pass |

The cost model can migrate into C++ later if desired.

## Expected effects

- **Removes recompile-dance hard-fails.** The `max_retry` `RuntimeError` path
  disappears: access that does not linearize is peeled, not retried-then-killed.
  This directly increases model coverage (the primary goal).
- **Removes the 4D rank cap.** Arbitrary-rank reshapes become expressible via the
  base-pointer loop; the `NotImplementedError` for >4D goes away.
- **Inverts the tile<->DMA dependency.** Tile size is chosen for compute / vlane
  efficiency only; the DMA conforms to whatever access results. No divisibility
  forcing, no oscillation. Tile selection simplifies.
- **Shrinks the codegen.** The `FloorDiv` / `ModularIndexing` recompile branches in
  `get_dma_info`, and the in-emission `RecompileSignal` paths, leave Python; the
  codegen emits one declarative op instead of procedurally forcing tiles.
- **Collapses two of the three padding sites for the DMA case.** Once divisibility
  is no longer required to represent access, the Python tile-adjust dance (1) is
  unnecessary for DMA, and `get_mask` (2) shrinks. `TestLoopPadding` (3) is
  addressed by Plan A. (Compute-side vectorization remainder is separate; see
  Plan A.)
- **Behavior-preserving for the common case.** Access without floor/mod still emits
  a single `dma_start` identical to today -> low-risk, incremental rollout.
- **Preserves the simulator contract.** The leaf is the existing customized
  `dma_start`; Spike / gem5 / TOGSim see the same descriptor kind, just more of
  them in a loop.
- **A clean tactical slice toward Plan B.** This factors out exactly the one piece
  that is actually broken (DMA decomposition) into a lowering pass, without the
  full linalg rewrite.
- **Cost-aware, so modeled performance is protected.** Peel small/outer, keep inner
  contiguous, relayout for pathological cases.

## Migration strategy

1. Define `togsim.transfer` (op + verifier) above the existing descriptor.
   Optionally formalize the descriptor as `togsim.dma_descriptor`.
2. Make the codegen emit `togsim.transfer` for loads/stores, carrying the access
   maps and vlane attributes it already computes.
3. Implement `decompose-transfer` with the fast path first (<=4D affine -> one
   `dma_start`), proving **bit-identical output** to today on a smoke test.
4. Add the peel path for floor/mod / >4D; validate end-to-end through all three
   simulators (the loop-of-descriptors must satisfy the TOG / Spike / gem5
   contract).
5. Add the relayout fallback gated by the cost estimate.
6. Remove the `get_dma_info` recompile branches once the pass covers their cases;
   use the failure ledger + assert-only `TestLoopPadding` to confirm nothing
   regresses before deleting.

## Relationship to Plan A and Plan B

- **Plan A (graph-level padding)** reduces how often peeling/relayout is needed by
  making dims granule-aligned, and retires `TestLoopPadding`. Complementary: this
  op makes representation robust; Plan A reduces constraint frequency.
- **Plan B (linalg)** is the full structured-ops rewrite; `expand_shape` /
  `collapse_shape` are the principled home for reshape, and the framework would
  generate the same peel/relayout under the hood. This transfer op is the narrow,
  now-achievable slice of that idea.

## Risks / open questions

- **C++ pass in the `PSAL-POSTECH/llvm-project` fork**: heavier iteration
  (rebuild), logic split across two repos. Mitigated by the hybrid split (smarts in
  Python, pass is mechanical).
- **TOG / Spike / gem5 contract on a loop of descriptors.** If TOG generation
  assumes "one DMA = one node," the loop form needs handling. Validate at step 4.
- **Cost model accuracy** for peel-vs-relayout; start with a simple
  descriptor-count threshold and refine against measured cycles.
- **Dynamic shapes**: `iter_bounds` as SSA operands and symbolic-divisor floor/mod
  (semi-affine) must be handled by the pass.
- **Relayout fallback** needs a scratch buffer and a copy kernel; account for its
  memory and cycle cost in the decision.
- **Async / tag management across the peel loop**: double-buffering / compute
  overlap must survive decomposition (e.g. keep the inner large DMA async, sequence
  the outer peel).
