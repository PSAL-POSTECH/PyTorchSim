# TOGSim C++ Trace Generation

TOGSim's timing path consumes a **Tile-Operation Graph (TOG)**: the stream of
modeled instructions a kernel executes (tile loads, tile computes, tile stores)
together with the dependency edges between them. The timing core turns that
graph into cycles.

This document describes how the TOG is produced: the kernel's post-vcix MLIR is
compiled into a small C++ program, and *running* that program emits the graph.
It replaces an earlier producer that materialized a flattened graph at compile
time and serialized it as ONNX.

Vocabulary used throughout:

- **Producer** — the generated `.so`. Shape-parametric C++ compiled from the
  kernel's MLIR. It contains no timing model and no functional compute; running
  it emits a trace and nothing else.
- **Trace record** — one `togsim_*` callback invocation, i.e. one modeled
  instruction.
- **Bridge** — the TOGSim-side code that turns the recorded trace into a
  `TileGraph` of `Instruction`s with dependency edges.
- **Core** — TOGSim's existing timing model (systolic arrays, VPU, DMA engine,
  SRAM, DRAM/NoC). It is unchanged by this pipeline.

## 1. Motivation

The legacy TOG producer (`MLIR -> Python dict -> ONNX -> C++ TileGraphParser`)
had four structural problems:

1. **"ONNX in name only."** The graph was serialized as ONNX, but every op was a
   custom `torchsim_*` attribute. That paid ONNX's costs (rigid schema,
   protobuf, stringly-typed attributes) for none of its interop value. The
   schema lived in three places — Python (`extension_op.py`), ONNX
   (`AsmParser/onnx_utility.py`), C++ (`TileGraphParser`) — and drifted.

2. **Synchronization was ad-hoc and DMA-specific.** One concept ("an async op
   completed; a consumer may proceed") was expressed two different ways: a
   content-addressed `tag_table` with overloaded magic values (`0` pending, `1`
   signaled, `>1` consumed-count, `-1` sparse) plus a separate
   `Instruction::ready_counter` / `child_inst` edge mechanism. Only the former
   worked for DMA.

3. **Static shape was baked in.** Loop bounds were resolved to constants and the
   graph fully materialized per shape, so dynamic shape forced a recompile per
   shape — pathological for LLM decode (a new `seq_len` every step) and MoE
   (variable expert load).

4. **Loop-flattening hacks.** `loop_end` tricks, the `calc_tag` content hash,
   dedup-by-skip and magic offsets existed only to flatten loop nests into a
   static graph.

See [Appendix A](#appendix-a-legacy-path-references) for the file references.

## 2. Model: trace-driven, not graph-materialized

Rather than materializing a flattened graph, **the TOG becomes a stream emitted
by running a shape-parametric producer.** The producer keeps loops as loops
(symbolic bounds become C++ function parameters) and calls a small set of
callbacks. Each call emits one trace record. TOGSim `dlopen`s the producer,
injects a callback context, and records the stream.

| Problem | Resolution |
|---|---|
| ONNX-in-name-only, 3-place schema | The C ABI is the single contract. No ONNX. |
| DMA-only, ad-hoc sync | Dependencies are explicit dataflow edges (§10). Async-DMA data arrival is one explicit barrier op keyed by the runtime tag slot. |
| Static shape | Loop bounds pass through from MLIR verbatim; symbolic bounds become native C++ loop bounds, so trip count is dynamic. |
| Loop-flatten hacks | Loops stay loops. `calc_tag` and dedup disappear. |

This is **not** a dynamic hardware scheduler. Control flow is still statically
emitted by the compiler; the `.so` is a deterministic *trace generator*, not a
timing model. The trace-as-data boundary is preserved, so the timing core is
untouched.

## 3. Trace op vocabulary

Four primitives. Everything else is composition.

- `dma(dir, arg_id, offset, shape, is_async, tag_id, tag_slot, read_bufs,
  write_bufs)` — `dir ∈ {LOAD, STORE}`. A **synchronous** DMA is blocking: it
  finishes when its data arrives, and consumers depend on it directly. An
  **async** DMA returns control immediately and signals its tag at data arrival
  (DMA response-complete); a later `memory_barrier` is the explicit point that
  waits on it.
- `compute(tile_id, compute_type, dims, read_bufs, write_bufs)` — one fixed-size
  tile kernel. Its cost is looked up (§6), never computed here.
- `memory_barrier(tag_id, tag_slot, write_bufs)` — the explicit async-DMA sync.
  It waits until the async DMA carrying the same `(tag_id, tag_slot)` has
  delivered its data, then becomes the last writer of the loaded buffer so
  consumers gate on arrival. It is the source IR's `togsim.wait` mapped through,
  not a synthesized barrier.
- `dispatch(tile_fn, iv, n_iv)` — run one parallel work-item (§9.3).

Control flow lives in the producer: ordinary `for`/`if` with runtime bounds.

Two things share the word "tag", and the pairing key is **both together**:

- **`tag_id`** — compile-time identity of a DMA's tag memref (which logical
  channel: the A-load vs the B-load).
- **`tag_slot`** — the runtime SRAM tile slot the loaded tile occupies (the
  double-buffer / SRAM-capacity index).

The key must include a runtime component: one static `togsim.dma` op executes
once per loop iteration, each iteration writing a different slot, so a
compile-time id alone cannot express the per-iteration pairing.

`lower_to_vcix` writes the wait's tag index with a `-acc_iv` term per
accumulation (reduction) loop var — a sentinel marking the reduction axis, not
an arithmetic offset — and `build_skeleton` strips those terms so a
`memory_barrier` waits on the same slot its async load wrote. (The legacy
`TileGraphParser` mirrors this by skipping stride `-1`.) Without the strip, the
producer evaluates `-acc_iv` to a negative slot at reduction iteration > 0 and
the pairing fails on subtile + multi-tile-K.

## 4. Decisions

| Axis | Decision |
|---|---|
| Input MLIR | Use the **given MLIR as-is**. Do not touch inductor, the MLIR templates, or shape plumbing. Whatever bounds the MLIR carries (const or symbolic) pass through verbatim. |
| MLIR -> C++ | **EmitC dialect + `mlir-translate --mlir-to-cpp`** (upstream). |
| `.so` <-> TOGSim | **`dlopen` + an opaque `EmitCtx` callback context.** The ABI boundary is the main design surface. |
| `.so` role | **Timing trace only.** Functional correctness stays on the Spike/LLVM path. Every op without a timing dependency is stripped; the loop skeleton, the API ops, and the ops feeding bounds/addresses remain. |
| Compute cycle | A separate annotation pass reuses **gem5 sample-mode** to build a precomputed `tile_id -> cycle` table, looked up at runtime. |
| Dynamic shape | Falls out of symbolic loop bounds. Per-tile cost is static (tiles are fixed-size); only trip count is dynamic. |

## 5. Architecture

### 5.1 Artifacts (per kernel)

- **Trace `.so`** — compiled from the skeleton MLIR. Shape-parametric: symbolic
  bounds become C++ function parameters.
- **Cycle table** — `tile_id -> (cycle, overlapping_cycle)`, a TSV sidecar.

Both are written next to the kernel's `tile_graph.onnx`. TOGSim picks the `.so`
up automatically; `TORCHSIM_LEGACY_TOG=1` forces the deprecated ONNX path.

### 5.2 Pipeline

```
post-vcix MLIR (affine/scf.for + togsim.transfer/wait + vcix/vector compute)
|
+-- Branch A (trace):
|     build_skeleton.py    loops kept, bounds as-is (symbolic preserved)
|                          togsim.transfer -> togsim.dma(..., tag_id, %tag[%idx], is_async)
|                          togsim.wait     -> togsim.memory_barrier(tag_id, %tag[%idx], write_bufs)
|                          compute body    -> togsim.compute(tile_id, dims)
|                          DCE everything with no path to a loop bound, an
|                          address, or an API operand
|     dep_analysis.py      per-op read/write SRAM buffer sets (SSA) + the vcix
|                          preload/matmul pairing (§10.2)
|     lower_to_emitc.py    togsim.* -> emitc.call_opaque; drive the upstream
|                          lower-affine / convert-{scf,arith,func}-to-emitc
|     mlir-translate --mlir-to-cpp   ->  C++
|     g++ -shared                    ->  trace.so
|
+-- Branch B (cost):
      cycle_table.py       reuse the gem5 sample-mode cycle_list already computed
                           in extension_codecache -> tile_id -> (cycle, overlapping)

TOGSim:
  run_producer()          dlopen(trace.so), resolve togsim_kernel, inject EmitCtx
                          { core pool; record sink; cycle table }, run it
                          -> a TraceRec stream
  trace_to_tilegraph()    TraceRec -> TileGraph (Instruction DAG, §10)
  Simulator / Core        cycles, DRAM traffic  (unchanged)
```

### 5.3 Components

- **`togsim_ops.py`** — the op vocabulary. The ops are kept *unregistered* (like
  the existing `togsim.transfer`), so no C++ dialect registration is needed and
  the togsim->emitc step is a Python rewrite rather than a registered
  ConversionPass.
- **`build_skeleton.py`** — reduces the kernel to loops + API ops. Preserves
  `is_async`; maps `togsim.wait` through to an explicit
  `togsim.memory_barrier`. The IR verifies across sibling prefetch/compute loop
  nests because the DMA/barrier pairing is by runtime tag slot, not a
  cross-region SSA edge.
- **`dep_analysis.py`** — the read/write SRAM buffer sets per op (§10.2).
- **`lower_to_emitc.py`** — rewrites `togsim.*` to `emitc.call_opaque`, outlines
  the work-item body (§9.4), then drives the upstream conversion passes.
- **`cycle_table.py`** — the `tile_id -> (cycle, overlapping_cycle)` sidecar.
- **`togsim_runtime.{h,cc}`** — the C ABI and the `EmitCtx` callback
  implementations.
- **`togsim_loader.h`** — `run_producer`: `dlopen`, ABI-version check, run,
  record.
- **`togsim_trace_bridge.{h,cc}`** — `TraceRec` stream -> `TileGraph`.

### 5.4 ABI (v12)

`mlir-translate --mlir-to-cpp` lowers `emitc.call_opaque` to *free function*
calls, so the contract is a set of `extern "C"` functions taking an opaque
`EmitCtx*` first argument. The loaded `.so` links back into the Simulator binary
(built with `ENABLE_EXPORTS`), so the symbols resolve without an explicit
function table. `togsim_abi_version()` guards against a producer built against a
stale header.

```c
typedef struct EmitCtx EmitCtx;
typedef void (*togsim_tile_fn)(EmitCtx*, int64_t* iv, int32_t n_iv);

int32_t togsim_abi_version(void);

void togsim_dma(EmitCtx*, int32_t dir, int32_t arg_id, uint64_t offset,
                int32_t ndim, const int64_t* dims, const int64_t* strides,
                int32_t elem_bits, int32_t is_async,
                int32_t tag_id, uint64_t tag_slot,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write);

void togsim_compute(EmitCtx*, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);

void togsim_memory_barrier(EmitCtx*, int32_t tag_id, uint64_t tag_slot,
                           const int64_t* write_bufs, int32_t n_write);

void togsim_dispatch(EmitCtx*, togsim_tile_fn fn, int64_t* iv, int32_t n_iv);

// entry point the loader resolves:
void togsim_kernel(EmitCtx*, int64_t* shape_args, int32_t n_shape_args);
```

`offset` is an **element** offset within tensor `arg_id`, computed by the
producer from the loop indices; only the runtime knows the tensors' allocation
bases, so it forms `base[arg_id] + offset * elem_bytes`. `compute_type` is
`0` vector / `1` matmul / `2` preload, and routes the op to the VPU or the
systolic array.

## 6. Compute cost model

A `togsim.compute(tile_id=...)` says *which* tile to compute, not how long it
takes. Because tiles are fixed-size (`TILE_M/N/K`), each tile's cost is
invariant — only the trip count varies with shape — so it is sampled once and
stored, keyed by `tile_id`. Two numbers per tile, mirroring the legacy TOG:

- `cycle` — full compute latency, sampled by gem5 sample-mode (the existing
  `cycle_list` measurement, reused so both paths stay cycle-consistent).
- `overlapping_cycle` — the portion that overlaps the previous instruction in
  the systolic pipeline. The timing core uses it as
  `finish = prev.finish + cycle - overlapping`. Derived exactly as the legacy
  path does: vector -> `0`, matmul -> `max(cycle - x_offset, 0)`, preload ->
  `max(cycle - w_offset, 0)`.

`togsim_compute` looks both up and sets them on the `Instruction`.

**Remainder tiles.** When a dimension is not divisible by the tile size, the
edge tile is partial and its true cost differs from the table entry. Today it is
charged the full-tile cost. Sampling a separate `tile_id` for the remainder is
the alternative; see §11.

## 7. EmitC lowering notes

`lower_to_emitc` drives the upstream `lower-affine`, `convert-scf-to-emitc`,
`convert-arith-to-emitc` and `convert-func-to-emitc` passes. One gap in this
LLVM 20 build: `convert-scf-to-emitc` emits `emitc.for` with `index` bounds, so
`convert-arith-to-emitc` leaves `builtin.unrealized_conversion_cast` on the
bounds (`emitc.size_t` <-> `index`) that `--reconcile-unrealized-casts` cannot
fold and `mlir-to-cpp` cannot print.

`_retype_for_to_size_t` therefore retypes each `emitc.for` to `!emitc.size_t`
bounds and induction variable, and folds the residual casts. A `size_t` IV also
makes the lowered *address* arithmetic cast-free, which is what lets each
`togsim_dma` pass a real `(arg_id, element offset)` computed from the loop IVs.

Unregistered ops (`togsim.*`) have no registered conversion patterns, which is
why the rewrite to `emitc.call_opaque` must be a custom pass and must run before
the upstream conversions.

## 8. Validation

The reproduction path for a single kernel:

```sh
python -m PyTorchSimFrontend.mlir.passes.lower_to_emitc <postvcix.mlir> \
    --so trace.so [--emit-cpp trace.cpp]
bin/Simulator --config <config.yml> --trace_so trace.so \
    [--cycle_table trace_cycles.tsv] [--log_level trace]
```

`--log_level trace` prints the per-instruction issue/finish timeline, which is
how the dependency model is checked: on a 256^3 GEMM the trace shows the
preloads and matmuls pipelining across the systolic arrays, the store waiting
for the last matmul to drain, and each compute waiting for its async weight
load's *data arrival* rather than its issue. Compute work and DRAM traffic match
the legacy path on the same gem5 cycle table.

## 9. Parallelism, reduction, and core dispatch

### 9.1 Where the semantics come from

Nothing has to be inferred. The post-vcix `affine.for` already carries the
mapping decision the frontend made, and `build_skeleton` preserves it:

| attribute | meaning | role |
|---|---|---|
| `outer_loop` | PARALLEL axis (e.g. GEMM m, n) | independent output tiles -> distributable across cores |
| `accumulation_loop` | REDUCTION axis (e.g. GEMM k) | partial sums into one output tile -> ordered dependency |
| `inner_loop` | tile micro-loop | within one tile |

### 9.2 Principle: bake intrinsic, parameterize extrinsic

Two kinds of hardware dependence must be treated differently:

- **Intrinsic** (vector lanes, `TILE_M/N/K`, systolic size) — defines the
  *content and cost of each instruction*. Already baked into the IR.
- **Extrinsic** (`num_cores`) — defines only the *distribution* of an otherwise
  fixed set of work-items. The tile set, the cost table, and the DMA tile shapes
  are all `num_cores`-invariant.

So `num_cores` is **not** baked into the producer. The producer is
**core-count transparent**: it never names a core or a core count.

### 9.3 Core-transparent work function + dispatch hook

The producer is two functions, split at the PARALLEL/ACCUMULATION boundary:

```c
// WORK: the trace for ONE independent output tile. Takes the PARALLEL indices;
// names no core. Reduction (k) is program order -- the accumulator is
// core-local, so the ordering is implicit.
static void togsim_kernel_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  int64_t mi = iv[0], ni = iv[1];
  togsim_compute(ctx, /*tile_id=*/0, ...);                  // acc init
  for (size_t ki = 0; ki < KT; ++ki) {                      // REDUCTION
    togsim_dma(ctx, LOAD, A, offA(mi,ki), ..., /*is_async=*/1, /*tag_id=*/0, ki%D, ...);
    togsim_dma(ctx, LOAD, B, offB(ki,ni), ..., /*is_async=*/1, /*tag_id=*/1, ki%D, ...);
    togsim_memory_barrier(ctx, 1, ki%D, ...); togsim_compute(ctx, 1, ...);  // preload
    togsim_memory_barrier(ctx, 0, ki%D, ...); togsim_compute(ctx, 2, ...);  // matmul
  }
  togsim_dma(ctx, STORE, C, offC(mi,ni), ...);
}

// DISPATCH: enumerate the PARALLEL domain, one work-item per call.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape, int32_t n) {
  for (size_t mi = 0; mi < MT; ++mi)
    for (size_t ni = 0; ni < NT; ++ni) {
      int64_t iv[2] = {(int64_t)mi, (int64_t)ni};
      togsim_dispatch(ctx, togsim_kernel_tile, iv, 2);
    }
}
```

Three orthogonal concepts:

- **Parallel** = each `togsim_dispatch` call is an independent work-item. TOGSim
  may place it on any core.
- **Reduction** = ordering *inside* one work-item: program order on its core.
- **Core assignment** = owned by `togsim_dispatch`, whose body lives in TOGSim.
  It round-robins a core from the partition's pool and brackets the call with
  `TILE_BEGIN`/`TILE_END` records, so the work-item's scope is exactly the
  function call. A core is an assignment, not a held resource; there is no free.

Because `togsim_dispatch` takes the work function as a pointer and forwards an
opaque `iv` array, one general dispatcher serves every kernel. The boundary
cannot be optimized away: TOGSim can only observe `togsim_*` callbacks across
the `dlopen` boundary, never a producer-internal call.

### 9.4 Codegen and ABI

`lower_to_emitc` outlines the innermost PARALLEL-loop body into an
`emitc.func togsim_kernel_tile(ctx, iv, n_iv)` and rewrites the dispatcher loop
to call `togsim_dispatch`. The `tile_id -> cycle` table is untouched by all of
this (it is `num_cores`-invariant).

### 9.5 Stance and the split-K exception

Refining "not a dynamic scheduler": **the per-work-item trace is static and
deterministic; only the work-item -> core binding is dynamic.** That is
independent-task distribution, not data-dependent control flow.

The transparent model holds while work-items are independent (data-parallel over
output tiles). **Split-K** — a reduction split *across* cores — breaks
independence: the producer would have to emit `c` partials plus a combine, so
the instruction stream would depend on `num_cores`, and the cross-core
dependency would have to be a real dataflow edge rather than program order.
Split-K is a deliberate, scoped exception, not supported today.

### 9.6 Work-items form a DAG

Work-items are not always a flat independent set. A computation *between*
parallel loops can only run once the inner parallel region completes:

```
parallel for m:
  parallel for n: A(m,n)     # leaf work-items, each writes a tile of m's buffer
  B(m)                       # join: reads that buffer -> depends on all n of this m
```

This needs **no new primitive**. It is the same dataflow-edge mechanism the
trace already uses (§10), at work-item granularity: the join op declares the
leaves' output buffer as an input, so the bridge makes it depend on every leaf
through the last-writer analysis.

The general picture: **work-items form a DAG whose edges are buffer
producer -> consumer dependencies.** Independent data-parallel work is the
degenerate edge-less case; barriers, reduction across a parallel axis, and
split-K are the same DAG with real edges.

### 9.7 Execution model: trace generation, not co-execution

The producer is a pure trace generator. It never computes cycles, models
hardware, or schedules. Two consequences pin the model:

- **What is an edge vs. what blocks.** Data dependencies are recorded *edges* —
  the producer does not block on them. The only thing that can ever block the
  producer is resource backpressure (finite cores, double-buffer slots, DMA
  queue depth), which is flow control, not timing semantics.
- **Cores, double-buffering, DRAM and NoC are the timing core's job** — reused,
  not reimplemented. The producer stays oblivious; depths and counts are
  consumer-side config.

Consumption is staged behind a swappable **sink**, so the choice touches neither
the producer nor the ABI:

| sink | threads | when |
|---|---|---|
| *materializing* (today) — callbacks append to a `TraceRec` vector, which the bridge turns into a `TileGraph` | none | static shape |
| *streaming* — callbacks push to a bounded queue; the producer runs as a fiber and blocks on backpressure while the DES loop advances time and resumes it | producer fiber | when dynamic-shape trace size makes full materialization impractical |

Even the streaming sink only blocks the producer on resource flow control, never
on timing-resolved data events. The single forward-compat requirement is that
the callback sink stays an interface.

## 10. Dependency model

The trace is an explicit dataflow DAG: every op declares the buffers it reads
and writes, and the bridge derives edges from that. There is no in-order chain,
no runtime content hash, and no op-pattern heuristic.

### 10.1 Representation

Each op carries the **SRAM buffer ids** it reads and writes (`read_bufs` /
`write_bufs`). The bridge maintains `writers(b)`, the set of current producers of
buffer `b`, and links each op against it. Resource scheduling — systolic-array
round-robin, double-buffering, SRAM capacity — stays entirely in the Core; the
trace only states producer -> consumer order.

### 10.2 Two dependency sources

A single "SRAM access" analysis is necessary but not sufficient:

| dependency | source | visible in SRAM? |
|---|---|---|
| load -> compute (a DMA writes X_spad/W_spad; preload/matmul read it) | SRAM last-writer per buffer | yes |
| the accumulator chain (init writes Y_spad; the epilogue read-modify-writes it; the store reads it) | SRAM last-writer on Y_spad | yes |
| **preload -> matmul** (a preload loads weights into the systolic array's registers; the matmul consumes them) | **the vcix opcode FSM** (op1 = preload pairs with the following op0 = matmul) | **no — SA-internal, not a memref access** |

On the 256^3 GEMM post-vcix, the SRAM buffers are `%0 = X_spad(A)`,
`%1 = W_spad(B)`, `%2 = Y_spad(acc)`. The matmul reads `%0` only; the preload
reads `%1`; the matmul does *not* read `%1`, because the weights come from the
systolic array. That is exactly why a memref-only analysis would let the matmul
run before the weight load. `dep_analysis` closes the gap by folding the
preload -> matmul pairing into a virtual `SA_WEIGHTS` buffer, so the FSM edge
becomes an ordinary last-writer edge.

Both sources are available *before* `build_skeleton` collapses the compute
bodies, which is why the analysis runs on the post-vcix IR.

### 10.3 Edge rules

An instruction has two completion points. A systolic-array op **occupies** its
unit for `cycle - overlapping_cycle` (the initiation interval) and its **result**
is ready at `cycle`. `DepEvent::ISSUE` releases a successor at the former,
`DepEvent::DONE` at the latter. The bridge applies one rule per buffer `b`:

- **Read `b`** — depend on every instruction in `writers(b)`. The edge is
  `ISSUE` when consumer and producer are both systolic-array ops (a matmul
  reading a preload or a matmul: they overlap on the pipeline), else `DONE`.
- **Write `b`** — replace: `writers(b) = {inst}`.
- **Exception — the commutative accumulator.** A matmul that both reads and
  writes `b` is accumulating (`Y += X @ W`). Skip its read edge, and on the
  write *union* rather than replace: it waits only for the non-matmul seed (the
  init or bias) and joins `writers(b)` without ordering against its
  co-matmuls. So the K matmuls do not chain through the accumulator, and a later
  reader joins all of them.

The last rule is what makes the store wait for the whole reduction while the
matmuls still pipeline: the store reads `Y_spad`, `writers(Y_spad)` is the union
of the K matmuls, and the store is not a systolic-array op, so it takes a `DONE`
edge from each. No explicit compute fence is needed.

### 10.4 Resource models, not edges

Write-after-read ordering is *not* expressed as an edge. Buffer reuse is a
resource question, and the Core already models it:

- **SRAM capacity.** A coarse tile is one *version* of its buffer; the fine DMAs
  that fill it share one allocation, freed once all of that version's consumers
  have issued. A buffer reused by the next reduction iteration or work-item is a
  new version that must wait for the old one to be freed — that is the
  double-buffer / WAR constraint, enforced by capacity rather than by ordering
  edges, so two versions may physically coexist.
- **Weight slots.** A preload takes a weight slot on a systolic array and holds
  it until its matmuls have consumed it, which caps how many preloads can be in
  flight per array.

A latency WAR edge would instead force the new load to wait for the old tile's
readers to *finish*, defeating double-buffering.

### 10.5 Async DMA: the memory barrier

An async DMA's own finish is *issue*-complete; its data arrives later, at DMA
response-complete. A raw last-writer edge on the DMA would therefore release a
consumer before the data exists — a real bug this model had to fix.

So an async load's last-writer edge is routed through its `MEMORY_BAR`. The DMA
registers its tag at issue; the barrier parks on `(tag_id, tag_slot)` in the
Core's existing tag table and is woken by `set_tag_finish` at response-complete;
the barrier then replaces `writers(b)`, so every consumer of the loaded buffer
gates on arrival. A synchronous DMA blocks to arrival itself and needs no
barrier.

The bridge mints a per-record unique key so that successive reduction iterations
of one static DMA op get distinct tag-table keys, and the matching barrier reuses
that key. This works because the recorded stream is already per-iteration — the
producer ran the loops.

## 11. Limitations and open work

- **Per-iteration tags are reconstructed in the bridge.** `dma_fine_grained`
  emits a fresh tag `memref.alloc` before each coarse load, but `build_skeleton`
  DCEs it and keys `togsim.dma` by the alloc's static identity, so the bridge has
  to re-derive per-iteration keys from record order. Threading the alloc identity
  through as an SSA tag handle would remove that.
- **Preload occupancy.** A preload's `overlapping_cycle` equals its `cycle`, so
  its occupancy is zero and concurrent preloads are not capped by the
  systolic-array count. Giving the preload a non-zero occupancy (the weight-load
  time) is a cycle-model input, not an edge-model change. This predates the trace
  pipeline and affects the legacy path equally.
- **Dispatch granularity.** Work-items are enumerated per innermost parallel
  loop. Distributing independent output sub-tiles across cores needs the sub-tile
  axis roles, which the inner loops do not currently carry.
- **Remainder tiles** are charged the full-tile cost (§6).
- **Op coverage.** The dependency model has been characterized in detail on
  GEMM. Other families go through the same rules but have not been
  cycle-characterized.
- **Dynamic shape.** Symbolic bounds survive the pipeline, but end-to-end
  `shape_args` plumbing and the streaming sink (§9.7) are not done.
- **Legacy path.** The ONNX TOG producer is deprecated and opt-in
  (`TORCHSIM_LEGACY_TOG=1`). It is retired once the trace path is stable across
  all op families. The cycle measurement (`cycle_list`, `x_offset`/`w_offset`) is
  shared, so both paths stay cycle-consistent meanwhile.

## Appendix A: legacy-path references

- `TOGSim/include/DMA.h` — `tag_table` (overloaded `0/1/-1/>1`) + `waiters`;
  `register_tag` / `set_tag_finish` / `register_tag_waiter` / `mark_tag_used`
  (= init / signal / wait / consume).
- `TOGSim/src/Core.cc` — the async-DMA signal path and the barrier wait/consume
  path over the tag table.
- `TOGSim/include/Instruction.h` — `ready_counter` / dependency edges and the tag
  fields.
- `PyTorchSimFrontend/mlir/passes/build_tog.py` — `TogBuilder.print_operation`
  dispatch and `_affine_for_bounds` (constant-bound resolution -> static shape).
- `AsmParser/tog_generator.py` — the ONNX serialization.
- `PyTorchSimFrontend/mlir/mlir_gemm_template.py` — the kernel template emitting
  the `affine.for` nest, `linalg.matmul`, and the `togsim.transfer` DMA ops.
