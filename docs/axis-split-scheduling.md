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

## Resolved (cont.)

- **`floor//1` / residual floor on multi-level split (fixed).** `simplify_with_ranges`
  cannot prove a *multi-term* numerator is below the divisor (e.g.
  `FloorDiv(z1 + 4*z2, 12)` with `z1<4, z2<3`), so a 3-level mixed-radix split left
  a residual floor that codegen rejected ("Not supporting this view operation").
  `_fold_with_ranges` now proves it directly from the split sub-var ranges via
  `bound_sympy`: `FloorDiv(num,d)->0` when `0<=num<d`, `ModularIndexing(num,k,m)->num//k`
  when `0<=num<k*m`. Fixes `rs3factor` (3-level chain `[1,4,12,24]`).
- **High-rank blow-up regression (guarded).** Splitting several axes can push the
  index rank past 4 (pixel_shuffle -> 5D), which triggers the nascent
  decompose-transfer peel + TOG path (see below). `find_split_plan` now has a rank
  guard: if applying the plan would make the index rank exceed 4, the whole plan is
  dropped and the kernel falls back to baseline. pixel_shuffle now passes (via
  baseline); 3D group_norm still splits (rank 4, allowed).

## Known issues / open

- **decompose-transfer peel <-> TOG incompatibility**: the >4D peel emits
  `memref.subview` + unrolled constant-offset `dma_start`, which the C++ TOG
  generation pass cannot read (empty `loop_idx_list`). The rank guard above
  side-steps it; the real fix is to rewrite the peel as an `affine.for` loop
  (keeping a loop index TOG can read) instead of unrolling. **Tracked as a GitHub
  issue + the `dma-transfer-lowering.md` TODO.**

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
- **Graph-copy for incompatible radices (case 5)** -- `graph_copy.py`,
  `TORCHSIM_GRAPH_COPY`. When two operands of an elementwise consumer carry
  incompatible-radix groupings on a shared axis (e.g. `a[c//2] + b[c%3]`, floor-by-2
  vs mod-by-3 on extent 6 -- not a divisibility chain), neither axis-split nor the
  recompile-dance can express it. We wrap the registered lowering entries (the
  make_pointwise results = every elementwise consumer, one place), trace each
  operand's loader with `extract_read_writes` to get its read indices, run the same
  `collect_boundaries` analysis, and if the union is not a chain, `realize()` the
  cheaper operand. realize() (not clone -- Inductor inlines clone, confirmed) forces
  a buffer: the consumer then reads it affine and the remaining single grouping is
  handled by axis-split. Validated: `incompat` (`a.repeat_interleave(2)+b.repeat(2)`)
  goes ERR -> allclose=True with `GRAPH_COPY+AXIS_SPLIT` (still ERR on default,
  confirming graph-copy is the fix); no regression on the pattern battery,
  test_add, resnet (compile overhead negligible).
- **Graph-copy for cross-axis floor/mod (case 7)** -- same hook. A transpose+reshape
  feeding a consumer that keeps the output dims separate (broadcast / softmax /
  layernorm / reduce-one-dim) produces a floor/mod whose argument spans *two* loop
  vars, e.g. `(3*p0+p1)//4`; axis-split cannot split a multi-var argument. We detect
  an operand whose read index has a floor/mod argument with >1 free symbol and
  replace it with `ExternKernel.copy_input` (a realized identity Pointwise). This is
  why copy_input and not `realize()`: `StorageBox.realize()` is a no-op on a
  ReinterpretView (a reshape), so it does not materialize view operands; copy_input
  forces the copy. The copy kernel iterates the operand's own contiguous shape, so
  its index collapses to single-var for axis-split, and the consumer reads the copy
  affine. Also covers single-operand consumers (a reduction reading a multi-var
  view). Validated allclose=True: reshape+broadcast, softmax(reshape),
  layernorm(reshape) (all ERR on default). NOTE the empirical correction: case 7 is
  NOT rare -- it is the common attention/norm "reshape then reduce/broadcast"
  shape; Inductor only avoids it when it can collapse the output to 1D (then the
  floor is single-var).

## Default-on + recompile-dance status

axis-split and graph-copy are **ON by default** (disable with `TORCHSIM_AXIS_SPLIT=0`
/ `TORCHSIM_GRAPH_COPY=0`). With them on, the codegen recompile-dance (tile-forcing
for floor/mod divisibility) is demoted from primary mechanism to a rarely-hit
fallback.

Measured under default-on (`TORCHSIM_RECOMPILE_LOG=1`), 33 tests, all pass:
- 16 core (elementwise/gemm/reduce/conv/view/fusion + mlp/resnet/transformer/vit): 0 recompiles.
- 7 broader families (cnn/pool/group_conv/sort/indirect_access/exponent/conv_fusion): 0 recompiles.
- 10 floor/mod patterns: 1 recompile total (an unrelated tile-divisibility in the
  3-level mixed-radix case).

**Full retirement of the dance is deferred** (it is still a real dependency, not
just a safety net): removing the floor/mod recompile branches would break the
3-level mixed-radix case (1 recompile) and any case axis-split/graph-copy do not
yet cover (case 6, >4D rank-guard skips). attention/sdpa families were not run here
(too slow locally) and need CI validation before retirement.

## Next steps

1. Eliminate the last recompile dependency (the 3-level mixed-radix sub-kernel) so
   the dance reaches 0/all -> then retire the floor/mod recompile branches (keep the
   non-floor/mod ones: non-power-of-2 vec size, indirect).
2. Graph-copy coverage: case 6 (non-dividing divisor / uneven cat -> pad or gather),
   and conflicts internal to templates (gemm/conv/sdpa).
3. High-rank interaction: cap split-induced rank or harden decompose-peel + TOG for
   high-rank tiles (pixel_shuffle end-to-end, #258).
4. Dynamic shapes -> symbolic divisibility / guards.
