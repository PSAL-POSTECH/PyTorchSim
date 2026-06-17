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

### Decomposition pass (contract): aligned-only mechanical peel

> **Scope decision (narrowed).** This pass is a **pure mechanical rank peel** of an
> already-affine access. It does **not** linearize floor/mod and does **not** do
> relayout. Those two responsibilities moved upstream (see "Division of labor"
> below): aligned floor/mod is removed by **axis splitting at the Inductor
> scheduling layer** (`axis-split-scheduling.md`), and misaligned access is
> resolved by **graph-level copy insertion**. So every `togsim.transfer` that
> reaches this pass is guaranteed per-axis affine; the only thing left is that its
> rank may exceed the 4D Gemmini descriptor.

The DMA descriptor is an **affine map of rank <= 4 with integer strides**
(`base + sum_i stride_i * idx_i`). The pass sees affine input (rank `D`) and:

1. **`D <= 4`** -> emit **one** customized `memref.dma_start`; the dims become the
   descriptor's <=4D shape/strides. Identical to today's output (fast path).
2. **`D > 4`** -> peel `D - 4` dims into an outer `affine.for` (marked `inner_loop`
   so the TOG pass reads the induction var); each iteration computes the DRAM base
   with one `affine.apply` (the peeled dims' linear contribution folded with the
   original index) and the **lane-banked physical** SRAM offset (dims outer than the
   vlane axis rescaled by the lane coeff -- the MVIN `block_stride` /
   `-dma-fine-grained` `buildSramAffineMap` rule, which needs the vector-lane count),
   delivered as the **last SRAM index operand**. The offset must go through the index,
   not a subview offset: the gemmini lowering reads the spad base via
   `extract_aligned_pointer_as_index`, which strips a subview offset.

That is the whole pass. There is **no linearization step** (upstream guarantees
affine) and **no relayout fallback** (upstream graph copy handles misalignment).

**Fail loud, not silent.** If the pass encounters floor/mod that does not reduce to
per-axis affine (misaligned), or a genuinely non-affine / indirect / gather index,
that is a **contract violation** -- upstream did not normalize it. The pass
**asserts/errors** rather than silently inserting a relayout. A silent in-pass copy
would be a hidden performance cliff and would duplicate, at the wrong layer, a
global layout decision only the graph can make correctly.

The decision point maps onto existing code: `get_dma_info` already raises at >4D.
That exact site becomes "emit `togsim.transfer`" (done, Phase 1), and this pass
consumes it. The recompile/tile-forcing dance is unnecessary because (a) aligned
floor/mod is gone before codegen and (b) the outer peel loop's `ceil` bound absorbs
non-divisible remainders.

### Division of labor (the affine-only contract)

| floor/mod source | handled by | cost | layer |
|---|---|---|---|
| aligned (single axis, divisor \| extent; group norm, broadcast) | axis split | free | Inductor scheduling |
| misaligned (uneven cat, non-factor reshape, multi-axis arg) | copy insertion | copy | FX graph |
| affine but rank > 4 (e.g. 5D permute) | mechanical peel | free | **this pass** |
| data-dependent / indirect / gather | indirect-indexing path | -- | out of scope |

Only the third row is this pass. The first two produce the affine-only invariant
this pass relies on.

### Relationship to memref-to-gemmini (ISA lowering) -- keep separate

`memref.dma_start` is the boundary, not the endpoint. The layering is:

    togsim.transfer  --[Python decompose]-->  memref.dma_start  --[Python lower_dma_to_gemmini]-->  Gemmini ISA

decompose-transfer stops at `memref.dma_start` and must **not** emit Gemmini
instructions directly; the ISA encoding is a separate pass
(`passes/lower_dma_to_gemmini.py`, which replaced the C++ test-memref-to-gemmini).
Rationale:

- **Separation of concerns**: decompose does descriptor decomposition (affine
  algebra: rank / peel); gemmini does instruction encoding (hardware). Different
  axes; merging couples affine logic with ISA detail. They stay distinct passes.
- **`memref.dma_start` is a shared contract** with multiple consumers
  (`lower_dma_to_gemmini`, `dma_fine_grained`, `build_tog` -- all now Python
  out-of-line passes; the C++ `-dma-fine-grained` / `-test-tile-operation-graph`
  are ported). Keeping it as the interface lets all of them stay unchanged.
- **gemmini is now a Python out-of-line pass too** -- the conversion-framework
  coupling (LLVMTypeConverter / getStridedElementPtr) was avoided by working at
  the memref level (`memref.extract_aligned_pointer_as_index` + arith for
  addresses, `llvm.inline_asm` for instructions; the existing standard lowering
  finalizes to LLVM). So both decompose and gemmini live in Python; mlir-opt keeps
  only the remaining custom passes.

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

(A pathological peel is not this pass's problem to fix: it means the operand's
layout is bad, which is a graph-level layout/copy decision, not an in-pass
relayout.)

### Placement: hybrid (least burden)

Keep the decision in Python (where shape/sympy info is available and iteration is
fast); keep the C++ pass purely mechanical.

| Step | Where |
|---|---|
| peel-plan decision (which dims to peel, count estimate) | Python |
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
  contiguous. Pathological layouts are fixed upstream (graph copy), not by an
  in-pass relayout.

## Migration strategy

1. Define `togsim.transfer` (op + verifier) above the existing descriptor.
   Optionally formalize the descriptor as `togsim.dma_descriptor`.
2. Make the codegen emit `togsim.transfer` for loads/stores, carrying the access
   maps and vlane attributes it already computes.
3. Implement `decompose-transfer` with the fast path first (<=4D affine -> one
   `dma_start`), proving **bit-identical output** to today on a smoke test.
4. Add the **affine** peel path for >4D; validate end-to-end through all three
   simulators (the loop-of-descriptors must satisfy the TOG / Spike / gem5
   contract). Make the pass **assert** on any non-affine residue (contract guard).
5. Land the upstream producers of the affine-only invariant: aligned axis split at
   scheduling (`axis-split-scheduling.md`) and misaligned graph copy insertion.
6. Remove the `get_dma_info` recompile branches once the pass + upstream cover their
   cases; use the failure ledger + assert-only `TestLoopPadding` to confirm nothing
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
- **Cost model accuracy** for the peel plan; start with a simple descriptor-count
  threshold and refine against measured cycles.
- **Dynamic shapes**: `iter_bounds` as SSA operands; affine peel must handle
  symbolic outer extents. (Symbolic-divisor floor/mod normalization is an upstream
  concern, not this pass.)
- **Upstream completeness.** The pass's fail-loud contract is only safe if the
  upstream producers (axis split + graph copy) actually normalize every misaligned
  case. Until they do, the assert may fire on real models -- track which ops trip it
  as the work-list for the upstream passes.
- **Async / tag management across the peel loop**: double-buffering / compute
  overlap must survive decomposition (e.g. keep the inner large DMA async, sequence
  the outer peel).

## Appendix: alignment theory (when floor/mod is statically decomposable)

This section records the math that decides, for a given DMA access, whether the
non-affine `FloorDiv`/`ModularIndexing` terms can be peeled into a *static* loop
of affine descriptors (free) or require a data movement (relayout / copy).

### Setup

A Gemmini-style descriptor addresses an element as

    addr(idx) = base + Σ_k stride_k · idx_k          (integer strides, rank <= 4)

i.e. each loop index `idx_k` contributes a **constant** stride. A DMA is
statically decomposable iff every index term it reads has constant stride over
the rectangular tile domain. Inductor index expressions, after fusion/view, carry
`FloorDiv(x, y)` and `ModularIndexing(x, y, z)` of the *flattened* loop variable
`x`. The question is when those reduce to constant-stride axes.

### Mixed-radix decomposition

Write the flattened index `x` (extent `E`) in mixed radix. For a `ModularIndexing`
with inner period `y` and modulus `z`, decompose uniquely as

    x = o·(y·z) + m·y + r,   with 0 <= r < y, 0 <= m < z, o >= 0

Then `FloorDiv(x, y) = o·z + m`, and `ModularIndexing(x, y, z) = m`. Each of
`o, m, r` is a separate **implicit axis** with a constant per-axis stride —
*provided the axis boundaries do not move across the tile*. That holds iff the
period divides the extent it partitions:

- `ModularIndexing(x, y, z)` is a valid rectangular axis  **iff  y·z | E**.
- `FloorDiv(x, y)` is a valid rectangular axis            **iff  y | E**.

**Aligned** = the divisor (and modular period `y·z`) divides the extent, so the
wrap point lands on a fixed axis boundary -> constant stride -> peelable for free.
**Misaligned** = the wrap point falls at a loop-value-dependent position inside the
descriptor (e.g. uneven `cat`, ragged split) -> the stride is not constant ->
**not** statically decomposable; only a relayout (physical copy) fixes it.

### One loop axis -> several implicit axes (complex fusion)

When fusion merges many dims into one flattened loop variable, a *single* loop
axis can expand into **several** implicit axes through nested floor/mod, e.g.

    x in [0, D0·D1·D2):
        a = FloorDiv(x, D1·D2)              # outer
        b = ModularIndexing(x, D2, D1)      # middle
        c = ModularIndexing(x, 1,  D2)      # inner

That is three implicit descriptor axes coming from one loop axis. This is the
general case the un-flatten must handle: it is **not** limited to splitting one
axis into two. Key consequences:

1. **The loop's own factorization is always aligned.** When the implicit axes
   come from re-reading the loop's *own* contiguous factorization (the common
   fusion case -- Inductor flattens contiguous dims then a consumer reads them
   back via floor/mod), every period divides by construction (`D1·D2 | D0·D1·D2`,
   etc.). So these un-flatten splits are **free** -- they just add descriptor
   axes, never a copy.
2. **Rank blows past 4 fast.** k implicit axes per loop axis, across multiple
   operands, means the descriptor rank exceeds the 4D Gemmini limit very quickly.
   This is exactly why `togsim.transfer` + the peel pass matters *more* under
   complex fusion, independent of any misalignment. The >4D branch in
   `get_dma_info` already routes these to `togsim.transfer`.
3. **Misalignment is still only from non-factor views.** An implicit axis is
   misaligned only when its period does not divide the extent -- i.e. the view
   does not factor along the loop's factorization (uneven `cat`, ragged split,
   group sizes that don't divide the channel count). Those, and only those, need
   relayout.

### Case-handling summary

| Source of floor/mod                            | Aligned? | Handling                          | Cost |
|------------------------------------------------|----------|-----------------------------------|------|
| Broadcast / dim-merge (`[N,1]->[N,M]`, `i//M`) | always   | un-merge (split loop axis back)   | free |
| Reshape along the loop's own factorization     | yes (`y·z\|E`) | un-flatten split, then peel for rank | free |
| >4D logical tile from complex fusion           | yes      | `togsim.transfer` -> peel into <=4D loop | free (extra DMA nodes) |
| Uneven `cat`, ragged split, non-dividing group | no       | graph copy insertion (relayout, upstream) | copy = TPU `concatenate` |

The TPU/XLA model is the reference: express only aligned views as
descriptor/bitcast (free reshape); never put a misaligned access in the
descriptor -- insert a copy (relayout) instead. Plan A (graph-level
force-contiguous / pad-to-granule, like XLA copy-insertion) is the upstream lever
that *reduces how often* the misaligned branch fires, keeping codegen affine-only.

## Implementation status (Phase 1: codegen emission)

Landed on branch `dma-transfer/codegen` (worktree), emission only -- the
decompose pass is deferred until explicitly signalled. A >4D access now emits a
`togsim.transfer` instead of hard-failing; without the pass it does not yet run
end-to-end (expected).

- **`mlir_common.py` `init_tile_size`** generalized to any rank. Logical tile is
  separated from the physical (<=4D) descriptor: only the innermost dims carry the
  vectorized tile, all further-outer dims stay 1, and there is no rank cap. The
  `nr_dim >= 3` formula reproduces the old 3D/4D values exactly (the old `[-4]=1`
  is subsumed by "outer dims stay 1"); scalar/1D/2D keep their special cases. This
  removes the old `raise NotImplementedError("dummy tile size fail!")` that
  conflated logical and physical tile rank.
- **`mlir_codegen_backend.py`**:
  - `__init__` adds `self._dma_needs_transfer = False`.
  - `get_dma_info` >4D `else` branch (was
    `raise NotImplementedError("Currently not implemented... ;)")`) now builds the
    full N-D tile (`set_tile_size`, vlane split/stride) and sets
    `self._dma_needs_transfer = True`.
  - `emit_transfer(...)` emits the generic-form `"togsim.transfer"(...)` op
    carrying `dma_kind`, `vlane_split_axis`, `vlane_stride`, `dram_stride`,
    `tile_stride`, `padding`, with operands `(dram, dram_idx, sram, 0, tag)`.
    `togsim` is an unregistered dialect, hence generic form.
  - `load()` (MVIN) and `store()` (MVOUT) check the flag: if set, reset it and
    call `emit_transfer`; otherwise the existing `get_dma_code` path is unchanged.
    So aligned <=4D DMAs are **bit-identical** to before; only >4D accesses change.

Validated: the 5D permute smoke test (`x.permute(4,3,2,1,0).contiguous() + 1.0`)
now emits MVIN/MVOUT `togsim.transfer` with 5D `dram_stride [1,6,30,120,360]` and a
`memref<1x1x2x4x2xf32,1>` tile, instead of crashing in `init_tile_size` or the
`get_dma_info` >4D branch.

### Phase 2: aligned-only peel pass (landed: unit-collapse path)

`passes/decompose_transfer.py` (registered in `passes/__init__.py`, runs before
`lower_vlane_idx`) lowers each `togsim.transfer` to a customized `memref.dma_start`:

- **Unit-dim collapse (done, validated).** Drop extent-1 tile dims so the
  descriptor reaches <=4D. The SRAM (spad) memref is collapsed to the effective
  rank via `memref.collapse_shape` (the customized `dma_start` convention requires
  SRAM rank == #indices == len(sram_stride)); DRAM stays flat rank-1 with its N-D
  structure in `dram_stride`. The `vlane_split_axis` is **remapped** from the
  original tile-dim index to the collapsed-dim index and rematerialized as a const
  (carried as a value attr precisely so the pass can remap it).
- Supporting changes: `emit_transfer` now carries the SSA operands a `dma_start`
  needs (`dma_type`, `vlane_stride`) + the `vlane_split_axis` value attr, so the
  pass is mechanical. `lower_to_llvm.py` gains `expand-strided-metadata` to lower
  `collapse_shape`.

Validated end-to-end (Gem5 + Spike + TOGSim, `allclose=True`) on the 5D permute
`x.permute(4,3,2,1,0).contiguous() + 1.0`; no regression on 2D/3D/elementwise.

- **Genuine >4 effective rank (affine.for peel; #258 resolved).** When >4 *non-unit*
  dims survive, the pass keeps the inner 4 as the <=4D descriptor and peels the outer
  dims into an `affine.for` nest (marked `inner_loop`), emitting one inner descriptor
  per iteration -- mirroring the `-dma-fine-grained` subtile loop. The DRAM base is
  `affine.apply(dram_idx + sum_k iv_k * dram_stride_k)` (one apply, not `arith.addi`,
  so the TOG pass walks the loop index through it). The SRAM slice offset is the
  **lane-banked physical** offset (split-outer dims rescaled by the lane coeff)
  delivered as the **last SRAM index operand**, *not* a `memref.subview` offset --
  `extract_aligned_pointer_as_index` in the gemmini lowering strips a subview offset,
  which is why the earlier full-unroll + subview attempt produced wrong data and the
  C++ TOG read an empty `loop_idx_list` (#258).

  The earlier full-unroll + subview form was isolation-only and INCOMPATIBLE with the
  TOG; the `affine.for` rework (this is exactly the #258 TODO) fixed both the TOG
  read and the numerics, so the axis-split rank guard was removed. Validated
  end-to-end (Gem5 + Spike + TOGSim, `allclose=True`) on `pixel_shuffle(x, 2) + 1.0`
  (5D tile) plus the gemm/bmm/conv/model suite.

The input stays per-axis affine by upstream guarantee. A non-affine residue is a
contract violation (aligned floor/mod
removal lives in `axis-split-scheduling.md`, misaligned relayout in graph copy
insertion -- see "Division of labor"); a genuinely non-affine / indirect index
would surface as a build failure here rather than being silently relaid out.
