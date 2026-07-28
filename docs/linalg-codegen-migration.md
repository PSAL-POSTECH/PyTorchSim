# Linalg-based codegen migration (Plan B)

Status: **deferred / design only**. This is not scheduled work. It records *why*
a linalg-based rewrite of the MLIR codegen will eventually be worth doing, and a
rough plan, so the decision does not have to be re-derived from scratch later.

For the near-term padding/dynamic-shape work, see Plan A (graph-level padding) in
the "Relationship to Plan A" section below. Plan A is the one to do first; it does
not depend on this document.

## TL;DR

The current MLIR codegen (`PyTorchSimFrontend/mlir/`) does not just emit loops —
it hand-implements the entire hardware mapping (tiling, vectorization, DMA,
scratchpad allocation, vector-lane distribution) as Python string emission. That
works, but it entangles three concerns that should be separable:

1. **what to compute** (the op math),
2. **how to map it onto the NPU** (tile sizes, vlane layout, DMA/scratchpad),
3. **how to make shapes fit the hardware** (padding / divisibility).

Concern 3 is currently spread across three places and is the source of the
"padding is heuristic and fragile" pain. Plan B factors concern 1 up to the
`linalg` dialect and rebuilds concern 2 as a set of MLIR lowering passes, so that
concern 3 falls out of the structured representation instead of being patched.

Plan B is a multi-month, higher-risk effort because the hardware mapping (concern
2) is bespoke and has no upstream equivalent. Do it when the payoff (separation,
reuse of upstream tiling/vectorization/fusion, easier autotuning, lower codegen
maintenance) is worth that cost — not as a means to fix padding alone.

## Where we are today

Entry point: PyTorchSim is an **Inductor backend**. Inductor handles capture,
decomposition, lowering to Inductor IR, scheduling, and **fusion**. Our code runs
at the codegen (step-5) stage and turns scheduled `SchedulerNode`s into MLIR.

`MLIRKernel` (`mlir/mlir_codegen_backend.py`) and friends emit, by hand:

- explicit DMA: `memref.dma_start` with MVIN/MVOUT encoding, `vlane_split_axis`,
  `vlane_stride` (`load`, `store`, `get_dma_code`);
- explicit scratchpad: `.spad` sections and `memref.global @bufN_spad`
  (`allocate_sram_buffer`, `get_scratchpad_buffer`);
- explicit vector-lane (vlane) distribution: `vmap.vlane_split_axis` /
  `vlane_stride`, `vlane_offset`, `get_used_vlane` (`_index_expr`,
  `get_dma_info`);
- explicit tile descriptors: `MLIRMultiDimTile` with per-axis tile sizes;
- explicit reduction loops: manual accumulator/iterator vars + `affine_yield`
  (`reduction`, `codegen_loops`);
- explicit vector-tail masking: `get_mask`.

This is roughly 5,500 lines across `mlir_codegen_backend.py`, `mlir_common.py`,
`mlir_template.py`, and `mlir_ops.py`, plus per-op templates
(`mlir_gemm_template.py`, `mlir_conv_*`, `mlir_bmm_template.py`,
`mlir_sdpa_template.py`, `mlir_sort_template.py`, `mlir_cat_template.py`,
`mlir_maxpool_template.py`). Most of it encodes how this specific NPU's memory
hierarchy and vector unit work. **That accumulated hardware knowledge is the asset
and the cost center for any rewrite.**

### Padding lives in three places

This is the key observation motivating the separation. Divisibility / padding is
handled by:

1. **Python recompile + tile adjustment** — `get_dma_info` FloorDiv /
   ModularIndexing handling, `_index_expr`, `convert_indirect_indexing`: "if the
   tile size does not divide the dim, bump the tile and raise `RecompileSignal` to
   recompile." This is the actual heuristic-padding body, and it is in Python, not
   MLIR.
2. **Python `get_mask`** — vector-tail masking for the innermost compute loop.
3. **MLIR `TestLoopPadding`** pass (in the `PSAL-POSTECH/llvm-project` fork) —
   rounds affine loop bounds up to a multiple of the step and resizes buffers,
   reverse-engineering the loop<->tensor mapping from affine maps.

Removing only (3) does not fix the fragility; (1) is arguably the larger source of
"the loop and the tensor do not line up." Any real fix has to address all three.

## Target architecture (Plan B)

Split the codegen into two layers with a clean boundary:

```
L0  ATen / FX logical graph (dynamic dims as SymInt)        <- Inductor, unchanged
L1  math layer:    Inductor IR -> linalg.generic / named ops (untiled, unvectorized)
                     = iteration domain + affine indexing maps + scalar body.
                       Decides nothing about tiles / lanes / DMA.
L2  mapping layer: tiling(+pad) -> vectorize(+vl) -> bufferize
                     -> DMA / scratchpad / vlane lowering -> leaf replacement
                     = hardware mapping, parameterized by a target description.
L3  LLVM / RVV / systolic microkernel
```

Why this helps:

- **The loop<->tensor mapping stops being reverse-engineered.** `linalg` ops carry
  `indexing_maps` + `iterator_types`, i.e. exactly the information `TestLoopPadding`
  tries to recover. Padding becomes a *parameter of the tiling transform*
  (`tensor.pad` generated with full knowledge of the maps), not a separate
  analysis pass.
- **The padding strategies we want become per-axis policy** in L2: systolic
  operand axes -> pad-to-uniform-tile (keeps a single 128x128 microkernel and a
  single gem5 latency entry); VPU / vector axes -> RVV `vl` (no padding); reduction
  axes -> `affine.min` clamp. One mechanism, selected per axis, instead of three
  scattered implementations.
- **Fusion policy stays in Inductor** (its scheduler decides what is one kernel),
  while the *mechanism* is upstream `linalg` tile-and-fuse. Pointwise epilogue
  fusion is essentially free because Inductor already composes `inner_fn`s into a
  single fused body -> one `linalg.generic`.
- **Dynamic shapes** are carried as `?` dims + symbolic affine, uniformly handled
  by L2 rather than by the recompile dance.
- **The cost model gets cleaner, not harder**: tile shape becomes an explicit
  attribute, so the gem5 latency table / TOG key on it directly instead of on
  inferred loop shapes.

### What is reusable vs bespoke

- **Reusable from upstream MLIR**: `linalg` ops, tiling + `tensor.pad`,
  vectorization, bufferization, the TilingInterface. The L1 translation
  (Inductor IR -> `linalg.generic`) is a *generic* translator (one path covers all
  regular pointwise/reduction), not per-op work — Inductor IR is already in
  structured iteration-domain + scalar-body form.
- **Bespoke, must be (re)written as MLIR passes**: MVIN/MVOUT DMA encoding,
  `.spad` scratchpad assignment, and the vlane_split mapping. **These have no
  upstream equivalent.** This is the bulk of the effort and the main risk: the
  knowledge currently in ~5,500 lines of Python emission must be re-expressed as
  custom bufferization-to-DMA / scratchpad / vlane-vectorization lowerings.

## Expressibility boundary

A regular (linalg.generic) op needs: a fixed rectangular iteration space; every
operand index an **affine** function of loop vars (no data-dependent indexing);
each axis purely `parallel` or a simple `reduction` (no scan/recurrence); a
statically-determined output shape (dynamic `?` ok, data-dependent shape not); and
a data-independent body (`arith.select` ok, data-dependent branching not).

Expressible: elementwise, broadcast, transpose, reductions (incl. multiple
reduction axes), matmul / bmm / contractions, direct conv, pooling, and fused
chains of these (matmul+bias+activation, prologue cast/dequant/transpose,
pointwise->reduce). `slice` / `pad` / `cat` are structured `tensor` ops (not
`linalg.generic`) but are supported by the same pipeline.

Not expressible -> stay as hand-written custom kernels: data-dependent indexing
(gather/scatter/embedding), sort/topk, data-dependent output shape
(nonzero/unique/masked_select), scan/recurrence (cumsum), and online/streaming
algorithms (flash-attention). In our op set this means **`sdpa` (online softmax)
and `sort` remain custom**; gemm/conv/bmm/maxpool are regular; `cat` is a
structured tensor op.

## Migration strategy (when Plan B is scheduled)

Incremental, op-by-op, with a numeric and a structural safety net. Do **not**
big-bang.

1. **Stand up the L2 pipeline for one op (matmul first).** Emit `linalg.matmul`
   from the matmul path; wire tiling(+pad, pad_value=0) -> bufferize -> a custom
   pass that lowers the 128^3 leaf tile to the existing systolic intrinsic ->
   LLVM. Milestone 1 is end-to-end correctness through all three simulators
   (Spike functional, gem5 latency, TOGSim cycle) for a single matmul.
2. **Demote `TestLoopPadding` to assert-only** (check, do not modify; fail/log if a
   loop bound is not a multiple of its step). Run the full test suite; anything it
   flags is a case L1/L2 has not covered yet.
3. **Migrate the remaining regular ops** (conv, bmm, pointwise, reductions,
   maxpool). Pointwise/reduction go through the generic L1 translator; VPU
   remainder via `vl`.
4. **Delete `TestLoopPadding`** once the assert-only version is silent across the
   suite, and retire the Python recompile/tile-adjust dance and most of `get_mask`.
5. Leave `sdpa` and `sort` as custom kernels that bypass L1/L2.

### Risks

- **Simulator-facing contract.** The current emission is tuned to produce exactly
  the LLVM / TOG shape the three simulators expect. `linalg`'s standard lowering
  emits different IR; re-validating the lowered artifact end-to-end (especially TOG
  generation, which may assume specific loop/memory patterns) is the real
  integration risk. This is why milestone 1 is "one matmul, end-to-end," not "all
  ops, emission only."
- **Re-encoding the hardware mapping.** DMA/scratchpad/vlane lowerings are new code
  with no upstream reference; budget for them dominating the schedule.
- **Inductor index expressions.** Inductor often collapses dims into one flat index
  with `FloorDiv` / `ModularIndexing`, which are not affine; `linalg` indexing_maps
  must be affine. Either keep dims uncollapsed or normalize div/mod back to
  multi-dim affine. (We already convert these to affine strings today in
  `_convert_sympy_to_mlir_expr`, but that path will need to be revisited for the
  map-carrying representation.)
- **Fusion seams.** Not everything fuses cleanly (reductions with mismatched axes,
  transpose/layout mismatches); expect some barriers, same as any framework.

## Relationship to Plan A (graph-level padding) — do this first

Plan A inserts padding at the FX/graph level (via Inductor's
`post_grad_custom_pass`) so that tiled dims arrive at codegen already aligned
(`tile_granule * symbol`). Under the hard constraint that we **keep the Inductor
spine**, Plan A is the high-ROI move:

- It collapses all three padding sites at once: the recompile/tile-adjust dance (1)
  becomes unnecessary (tiles always divide), `get_mask` (2) becomes trivial (no
  tail), and `TestLoopPadding` (3) becomes unnecessary.
- It does **not** touch the bespoke DMA/scratchpad/vlane mapper.

Key correctness facts that make Plan A tractable:

- Matmul contraction (K) padding with zeros is *exact* (additive identity); weights
  are constants, so they can be zero-padded once, offline, at no runtime cost.
- Padding only ever corrupts results when a *non-contraction* padded axis is later
  reduced (softmax over keys; layernorm if hidden is padded). Those points need
  masking; everything else is pad-transparent.
- Safety rule: default any op to slice-back-to-real-shape; only opt an op into
  "propagate padded shape" once it is proven pad-transparent or given a mask
  handler. Correct-by-construction; unknown ops cannot silently corrupt.

Plan A and Plan B are compatible: Plan A's graph-level alignment makes the eventual
Plan B simpler (L2 tiling rarely needs to pad, because dims already divide).

## Open questions

- Does the current toolchain (the `PSAL-POSTECH/llvm-project` fork) already ship the
  `linalg` + transform/tiling passes, or were they stripped? (Almost certainly
  present if it tracks upstream — verify before committing.)
- Can the systolic leaf be expressed cleanly as a match-and-replace on a fixed-size
  `linalg.matmul`, or does weight-stationary loading order force a more custom
  representation?
- How much of `mlir_ops.py` (the scalar `OpsHandler`) survives? It currently emits
  *vectorized* ops (compute_vec_size, broadcast) and is therefore entangled with
  vlane; the linalg body should be scalar, with vectorization done in L2.
