# TOGSim C++ Trace Generation — Status Report

Branch: `feature/togsim-cpp-trace`. Design of record: `togsim_cpp_trace.md` (esp.
§9); continuation notes: `togsim_cpp_trace_HANDOFF.md`. This file is a snapshot of
progress.

## 1. Goal

Replace the timing-path TOG producer (`MLIR -> Python dict -> ONNX -> C++
TileGraphParser`) with a compiled, shape-parametric trace producer
(`MLIR -> skeleton -> EmitC -> C++ -> .so`). TOGSim's timing core is preserved;
only the producer of its input changes. The key idea: do not flatten the TOG;
instead **run** a compiled C++ producer that emits the trace as a stream of API
calls.

Each API call emits one trace record = one modeled instruction, fed to the
existing timing Core. Dependencies are an explicit dataflow DAG (SRAM
last-writer per buffer + the vcix preload/matmul FSM). An asynchronous DMA is
synced to the consumer of its data by the **runtime tag slot** `(tag_id,
tag_slot)` through an explicit `togsim.memory_barrier` (ABI v11). An earlier
design used a compile-time `event_id` / event handle with `wait`/`signal`; that
was removed because one static DMA op runs once per loop iteration into a
different tag slot, which a single compile-time id cannot pair per iteration.

## 2. Pipeline

```
post-vcix .mlir (torch.compile output)
  | build_skeleton.py + dep_analysis.py (P1)  keep loops;
  |   memref.dma_start -> togsim.dma(tag_id, %tag[%idx], is_async, read/write_bufs);
  |   memref.dma_wait  -> togsim.memory_barrier(tag_id, tag_slot, write_bufs);
  |   compute block    -> togsim.compute; DCE the rest
  v
skeleton+API MLIR
  | lower_to_emitc.py (P2/C4)  togsim.* -> emitc.call_opaque; ABI signature; drive upstream
  |                            lower-affine/convert-*-to-emitc; _retype_for_to_size_t fixups
  v
EmitC --mlir-translate--> C++ --g++ -shared--> trace.so
                                                 | TOGSim loader (C6): dlopen + EmitCtx callbacks
                                                 v
                                       TraceRec stream (materializing sink)
                                                 | togsim_trace_bridge.cc -> existing Core timing
                                                 v
                                       cycles / DRAM traffic (real Core)
```

Side artifact: cycle table `tile_id -> (cycle, overlapping_cycle)` (cycle_table.py).

## 3. Milestones

| | State |
|---|---|
| P0 ABI header + togsim vocabulary | DONE (ABI evolved to v11) |
| P1 build_skeleton | DONE, verified (compute/dma/barrier match legacy TOG) |
| P2 lower_to_emitc -> .so | DONE (real GEMM .so built and run) |
| P3 loader/runtime + cycle table + real-Core run | DONE (runs end-to-end through the real Simulator/Core; below) |
| P4 symbolic/dynamic shape, streaming sink | TODO |
| P5 op-family migration (conv/SDPA/vector) | TODO |

P3 detail:

| | State |
|---|---|
| ABI (core_alloc, runtime tag pairing, dma address) | DONE (v11) |
| work-item boundary (togsim_core_alloc) | DONE |
| real tile DRAM addresses (approach A) | DONE, verified on 1024^3 |
| cycle_table builder (cycle + overlapping) | DONE |
| async DMA <-> consumer sync (runtime tag slot, memory_barrier) | DONE |
| explicit dataflow DAG (read/write_bufs last-writer) | DONE |
| C6 runtime + dlopen loader (materializing) | DONE |
| TraceRec -> existing Core timing feed | DONE (runs end-to-end through real Core) |
| cycle comparison vs build_tog (real gem5 table) | DONE: trace 2518 vs legacy 2698 |
| SRAM tile lifecycle / preload-occupancy refinements | partial (see §7) |

### TraceRec -> Core: now running end-to-end

`TOGSim/src/togsim_trace_bridge.cc` (`trace_to_tilegraph`) + a `--trace_so` mode
in `main.cc` feed the recorded trace into the REAL Simulator/Core. The producer
`.so` is `dlopen`'d (the Simulator is built with ENABLE_EXPORTS so the `.so`
resolves the `togsim_*` callbacks back into the binary), its trace recorded, then
bridged to a `TileGraph`: one `TileSubGraph` per work-item (core_alloc marker)
bound to its core, one `Tile` of MOVIN/MOVOUT/COMP/MEMORY_BAR/COMPUTE_BAR
`Instruction`s. Dependency edges are built by **last-writer per SRAM buffer**
(`read_bufs`/`write_bufs`); an async load's last-writer is the MEMORY_BAR paired
to it by the runtime `(tag_id, tag_slot)` (so a consumer waits actual data
arrival), and a COMPUTE_BAR drains the systolic-array pipeline before a store.
Build it (`cd TOGSim/build && cmake .. && make`) and run:
`bin/Simulator --config <yml> --trace_so gemm_trace.so`.

### Cycle comparison vs legacy build_tog (256^3 GEMM, real gem5 table)

Ran the same kernel through the legacy path (torch.compile -> gem5 -> build_tog
-> Simulator) and the trace path (the same post-vcix IR -> trace .so + the SAME
gem5 cycle_list -> --trace_so), both through the REAL Core. extension_codecache
has an opt-in TORCHSIM_DUMP_TRACE_SO=1 hook that dumps trace.so + trace_cycles.tsv
from the same cycle_list/offsets (best-effort, never breaks the legacy path);
compute-unit routing uses compute_type and the tag key uses a per-tensor addr_id
(set_addr_name(arg_id)+prepare_tag_key) so A and B don't collide on tag_slot 0.

**Result: the trace path totals 2518 cycles vs the legacy path's 2698 on the
same gem5 cycle table.** All togsim python tests pass; TOGSim builds. Compute
work and DRAM traffic match; the remaining difference is scheduling (the
explicit dataflow DAG plus the occupancy/latency SA-pipeline model overlap
differently than legacy's per-iteration BARs).

## 4. Components

- `build_skeleton.py` + `dep_analysis.py` — in-place reduction of post-vcix to
  "loop skeleton + togsim.* API"; `memref.dma_wait` mapped through to an explicit
  `togsim.memory_barrier`; read/write SRAM buffer ids attached; reuses legacy
  `TogBuilder` traversal.
- `lower_to_emitc.py` — skeleton -> EmitC by driving the upstream conversion
  passes plus `_retype_for_to_size_t` (clears residual index<->size_t casts).
  `togsim_dma` carries `(tag_id, runtime tag-index, is_async, read/write_bufs)`
  and returns void; `togsim_memory_barrier` carries `(tag_id, tag_slot,
  write_bufs)`; `togsim_core_alloc` inserted at the work-item boundary.
- `cycle_table.py` — `tile_id -> (cycle, overlapping)`, overlapping
  `= max(cycle - offset[type], 0)` (legacy formula); JSON sidecar.
- `TOGSim/src/togsim_runtime.cc` + `TOGSim/include/togsim_loader.h` — C6 runtime
  and `run_producer` (dlopen -> togsim_emit -> records TraceRec). dma resolves
  `base[arg] + offset*elem_bytes` and signals its tag at data arrival; the
  matching memory_barrier waits the `(tag_id, tag_slot)`; compute looks up the
  cycle table; core_alloc round-robins a runtime core pool.
- `TOGSim/src/togsim_trace_bridge.cc` — bridges the recorded TraceRec stream into
  the existing `TileGraph`/`Instruction` form for the real Core.
- `TOGSim/include/togsim_runtime.h` — producer ABI v11.

## 5. Locked design decisions

1. **Trace is a DAG, not a time order.** The consumer (existing Core) schedules
   per-core timelines from: op kind -> hardware unit, SRAM-buffer last-writer ->
   data dependency, same-core -> serial (reduction accumulate), SRAM slot ->
   capacity. Emission order != execution order.
2. **Async-DMA sync = runtime tag slot.** A `togsim.dma` carries `(tag_id,
   tag_slot)`; the matching `togsim.memory_barrier` (lowered from the source
   `memref.dma_wait`) waits on the same pair through the existing Core tag table
   (`prepare_tag_key`/`set_tag_finish`/`register_tag_waiter`). The DMA signals at
   data arrival; the barrier becomes the loaded buffer's last-writer so consumers
   gate on arrival. A sync DMA is blocking (no barrier). This replaced an earlier
   `event_id` / heap event-handle design, which could not pair a DMA op with its
   wait per loop iteration (one static op, a different tag slot each iteration).
   No `calc_tag` content-hash, no magic values, no FIFO.
3. **Core = runtime allocation.** `togsim_core_alloc` returns a core id (no free).
   `num_cores` is never baked into the producer -- it is the runtime pool size.
   A work-item's reduction stays on one core (sticky); different work-items get
   different cores -> multi-core.
4. **Intrinsic baked / extrinsic parametric.** vlane / tile sizes / systolic
   define instructions (baked); num_cores only distributes (runtime).
5. **Execution model:** P3 materializing (run producer to completion -> record ->
   feed existing Core); P4 streaming (coroutine, alloc-blocks on resources).
6. **Double-buffer = resource constraint.** Producer emits everything (no skew);
   capacity is the consumer's throttle. Requires SRAM tile lifecycle
   (alloc/free) in the trace -- the currently missing piece.

## 6. Verification (reproducible)

- togsim python tests pass: skeleton (contract + fixture), emitc (build + dlopen
  run), cycle_table, runtime. TOGSim builds.
- 256^3 GEMM: core_alloc -> dma(tag_id, tag_slot) -> memory_barrier(tag_id,
  tag_slot) -> compute; addresses A/B/C resolved (offset 0, single tile).
- 1024^3 GEMM: per-tile addresses correct (A[m,k]=m*1024+k -> 0,256,512;
  B[k,n]=k*1024+n -> 0,262144,524288).
- End-to-end through the real Core (256^3 GEMM, real gem5 table): trace 2518
  cycles vs legacy 2698.
- Legacy ONNX-TOG path untouched (comment-only diff), marked DEPRECATED, kept as
  the comparison reference.

## 6b. Reference timer (early sanity check; superseded by the real Core feed)

`togsim::simulate(RunResult, TimingParams)` (togsim_runtime.cc) was an early
standalone scheduler that timed the recorded TraceRec to prove the stream is
sufficient to be timed: per core a DMA-engine timeline (DMAs serialize, overlap
compute), a compute timeline (serial = reduction accumulate, with the `finish =
prev.finish + cycle - overlapped` pipeline overlap of Core.cc), and data deps.
It is NOT the production Core (no DRAM/NoC/L2 contention). It has since been
superseded: the recorded stream is now bridged into the real Tile/TileGraph ->
Core (see §3, and the 2518-vs-2698 result above). Retained here as context.

## 7. Remaining work (priority order)

1. DONE. Map TraceRec -> existing TOGSim Core Instructions (Tile/TileGraph,
   compute_cycle+overlapping, dataflow-buffer deps + runtime-tag barriers) and
   run through the real Core. Result: trace 2518 vs legacy 2698 on the same gem5
   table.
2. SRAM tile lifecycle in the trace (double-buffer throttle). togsim_dma carries
   `tag_slot` (the lowered SRAM tag index = the slot key the existing Core's
   Instruction.tag_idx needs); 0 for single-buffer kernels. Remaining: the
   consumer must use it to throttle in-flight loads to the buffer depth. The
   SRAM-buffer key is effectively (arg_id, tag_slot) since each load's DRAM
   tensor maps to its spad.
3. Preload concurrency cap / preload occupancy (design doc §10.5): give a preload
   a non-zero occupancy so concurrent preloads are capped at the SA count.
   Pre-existing in BOTH paths.
4. (later) deeper double-buffer pipelines (more tag slots), two-function outline,
   P4 streaming, symbolic shape, P5 op coverage (conv/SDPA/vector).

## 8. Risks / open

- SRAM lifecycle (double-buffer throttle) not yet implemented -- central to
  double-buffer/capacity accuracy on multi-tile kernels.
- LLVM 20 emitc constraints absorbed: emitc.for index bounds; old
  subscript-returns-element model; arith.divui/remui not lowerable -> core id is
  a runtime allocation (which became a design improvement).

### Explicit dataflow-edge dependency model: implemented

The dependency model is an explicit dataflow DAG, not in-order or runtime-tag
content-hashing. `togsim_dma`/`togsim_compute` carry read_bufs/write_bufs (SRAM
buffer ids; a virtual SA_WEIGHTS buffer folds the preload->matmul edge).
dep_analysis + build_skeleton attach them; lower_to_emitc emits them; the runtime
records them; the bridge builds the Instruction DAG by last-writer per buffer,
scoped per work-item. The one runtime-paired edge is the async-DMA data wait,
routed through an explicit `togsim.memory_barrier` keyed on `(tag_id, tag_slot)`
(see design doc §10.7.4). The systolic-array pipeline uses the occupancy/latency
split (§10.7), so accumulating matmuls pipeline rather than serialize.

Net (256^3 GEMM, real gem5 table, real Core): trace 2518 vs legacy 2698.
Per-output-tile dispatch for multi-core distribution is the next refinement
(today one dispatch per work-item).
