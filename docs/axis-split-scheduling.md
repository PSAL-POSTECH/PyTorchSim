# Aligned axis splitting at the Inductor scheduling layer

Status: **prototype / proposed**. Companion to `dma-transfer-lowering.md`. This doc
covers the *upstream* half of the affine-only contract: removing aligned
`FloorDiv` / `ModularIndexing` from index expressions before they reach MLIR
codegen, by splitting loop axes at the Inductor scheduling layer.

## Goal: the affine-only contract

We want the MLIR codegen (`get_dma_info` in `mlir_codegen_backend.py`) to receive
only per-axis affine index expressions:

    off(i,j,k,...) = base + Sum_k stride_k * loop_var_k     (stride_k constant int)

with **zero** `FloorDiv` / `ModularIndexing`. If that invariant holds, codegen no
longer fights non-affine indices: the recompile dance (RecompileSignal, forced
tile sizes, max_retry_compile), the heuristic `TestLoopPadding` pass, and the
hard-fail-on-conflict path all become unnecessary. Codegen's only remaining job
is the *mechanical* rank<=4 peel for the Gemmini descriptor (orthogonal; see
`dma-transfer-lowering.md`), which operates on already-affine input.

Two tools produce this invariant, matching the alignment theory:

- **aligned floor/mod -> axis split** (this doc): loop transformation, free, no
  data movement.
- **misaligned floor/mod -> graph copy insertion** (XLA-style): genuine data
  movement; out of scope here.

"Aligned" means the floor/mod argument is a *single* iteration variable `v` of
extent `E` and the divisor `k` (resp. `k*m` for ModularIndexing) divides `E`, so
splitting `v = outer*k + inner` lands the wrap point on a fixed axis boundary.

## Where: the scheduling layer already rebuilds LoopBody

`mlir_scheduling.py` already does loop-IR surgery at the scheduling layer:

- `revert_group` (line ~219) rebuilds a `LoopBody` from `get_store_function()`
  with a chosen `var_ranges` -- it undoes Inductor's `simplify_and_reorder`.
- `codegen_node` (line ~246) injects dummy size-1 loops when Inductor
  over-simplified the group.

Axis splitting is the same operation with a different `var_ranges`: split the
axes carrying aligned floor/mod, then rebuild. No new infrastructure -- reuse
`LoopBody`. This is "upstream" of MLIR codegen and native to Inductor's IR (sympy
ranges + index exprs), so we are not reverse-engineering MLIR text.

## How: detect / rebuild / hook

Implemented in `PyTorchSimFrontend/mlir/axis_split.py`, wired into
`codegen_node` behind `TORCHSIM_AXIS_SPLIT=1` (dump with
`TORCHSIM_DEBUG_AXIS_SPLIT=1`).

1. **Detect -- `find_split_plan(nodes)`**: scan each node's
   `_body.indexing_exprs` for `FloorDiv(v, k)` / `ModularIndexing(v, k, m)` where
   `v` is a single iter var and the divisor divides `v`'s extent. Return
   `{axis_index: divisor}`, keyed positionally so it applies to every fused node
   sharing the iteration space.
2. **Rebuild -- `build_split_body(node, plan)`**: rebuild `node._body` /
   `_sizes` with the split var_ranges; feed the store function the index
   expression `outer*k + inner` at the split dim so the floor/mod collapses.
3. **Hook -- `codegen_node`**: apply the plan to every node
   (`_sizes, _body, group = ...`), then recompute the group.

## Empirical validation (group norm)

`group_norm(x[2,6,4,4], num_groups=3)` normalize kernel, before vs after split:

    before  var_ranges={p0:2, p1:6, p2:16}
      idx0 = 96*p0 + 16*p1 + p2          # x input, affine
      idx1 = 3*p0 + (p1//2)              # mean/rstd  <- FloorDiv(p1,2), 2|6 aligned
      idx2 = p1                          # weight/bias, affine

    after   plan={1: 2}, var_ranges={s0:2, s1:3, s2:2, ...}
      idx1 = 3*s0 + (s1//1)             # FloorDiv collapsed to identity -> s1
      ...                               # mean now affine; s2/spatial broadcast (stride 0)

The FloorDiv is eliminated. group `(2,6,16) -> (2,3,2,...)`.

## Coverage (what this framework can and cannot do)

| Case | Example | Status |
|---|---|---|
| aligned FloorDiv, single var | group norm `c//2` (2\|6) | DONE (prototype) |
| aligned ModularIndexing | `(v//k)%m`, k*m\|E | needs mixed-radix multi-split |
| multiple radices on one axis | `//2` + `%3`, E=6 | needs nested split (now: first divisor only) |
| reduction-axis floor/mod | `r//k` inside reduce | needs reduction-var splitting |
| divisor does not divide extent | C=8 groups of 3; uneven cat | IMPOSSIBLE by split -> graph copy |
| multi-axis argument | `(4p+q)//6` non-factor reshape | IMPOSSIBLE by split -> graph copy |
| dynamic / symbolic | `v//s`, symbolic extent | separate symbolic/guard path |

The aligned class is the framework's domain (currently only single-split
FloorDiv); the misaligned class is structurally a graph-copy problem.

## Resolved

- **5D blow-up (fixed).** `build_split_body` now reindexes the already-collapsed
  `node._body` via `LoopBody`'s copy path (pass the body as `fn` ->
  `_init_with_copy`), instead of re-tracing the raw store function over
  `inode.data.get_size()`. This keeps merged dims merged (spatial stays `16`),
  so group_norm goes `(2,6,16) -> (2,3,2,16)` (4D, no cap hit), and
  `_init_with_copy`'s `simplify_with_ranges` folds the split floor.
- **`floor//1` residue (fixed).** The fold only happened once the new symbols
  carried integer/non-negative assumptions: build them with
  `torch._inductor.utils.sympy_index_symbol` (not bare `sympy.Symbol`), which is
  also why the index prefix must not be `s` (reserved for shape symbols). With
  this, `idx1 = 3*p0 + (p1//2)` becomes `3*z0 + z1` -- the channel FloorDiv is
  gone, not left as `z1//1`.
- **Symbol conventions.** Index dims use the `z` prefix; reduction dims use the
  `r` prefix and are kept after the index dims so the reduction axis stays
  innermost (`var_ranges` is ordered iter-then-reduce; `LoopBody.sizes` splits on
  `len(iter_vars)`). LoopBody var names are remapped to `index<N>` during MLIR
  codegen, so the prefix is internal -- but it must not collide with the original
  body's names (those are `p`/`q`, so `z`/`r` are safe).

## Known issues / not yet exercised

- **Incompatible radices**: if an axis carries radices that do not form a
  divisibility chain (e.g. floor-by-2 and mod-by-3 on extent 6), the axis is left
  unsplit (its floor/mod falls back to the recompile path). A single mixed-radix
  split cannot linearize incompatible radices.
- **High-rank blow-up downstream**: splitting several axes can push the iteration
  rank past 4 (e.g. pixel_shuffle -> 5D tile), which then exercises the
  decompose-transfer peel and the TOG serialization on high-rank tiles. The
  linearization is correct, but those downstream paths are nascent (one peel
  subview bug fixed here; TOG `loop_idx_list` on high-rank tiles still open).

## Done

- **Mixed-radix (ModularIndexing + multi-radix)**: `find_split_plan` returns a
  per-axis divisibility-chain of boundaries; `build_split_body` splits into one
  sub-var per segment (`v = sum_i d_i*b_i`). Validated allclose=True on group_norm
  (FloorDiv, `[1,2,6]`) and `x.repeat(1,2)` (single-axis ModularIndexing,
  `[1,8,16]`); pixel_shuffle (floor+mod on two axes) linearizes correctly.
- **Reduction pass-through**: reduction dims keep the `r` prefix and stay innermost
  (after the index dims). Exercised via the `TORCHSIM_AXIS_SPLIT_FORCE` validation
  gate (force-split a reduction kernel's index axis even without floor -- an
  identity transform, so allclose must hold): layernorm `(512)->(256,2)` and
  reduce `(68)->(34,2)` keep their reduction groups and pass.

## Next steps

1. Misaligned cases -> graph-level copy insertion (separate work).
2. High-rank interaction: decide whether to cap split-induced rank or harden the
   decompose-peel + TOG path for high-rank tiles (pixel_shuffle end-to-end).
3. Dynamic shapes -> symbolic divisibility / guards.
4. Turn axis-split on by default for covered cases; retire the matching
   recompile-dance branches; measure coverage.
