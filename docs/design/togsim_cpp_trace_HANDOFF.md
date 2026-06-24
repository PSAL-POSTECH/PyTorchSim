# Handoff — TOGSim C++ Trace Generation

Continuation notes for picking this work up in a fresh session. Read alongside
the full design: [`togsim_cpp_trace.md`](./togsim_cpp_trace.md) and the snapshot
[`togsim_cpp_trace_STATUS.md`](./togsim_cpp_trace_STATUS.md).

## Goal (one line)

Replace the timing-path TOG producer (MLIR -> Python-dict -> ONNX -> C++ parser)
with a compiled, shape-parametric trace producer (MLIR -> EmitC -> C++ -> `.so`);
TOGSim's timing core is preserved.

## Current state (one paragraph)

The trace pipeline is implemented end-to-end and runs through the REAL
Simulator/Core on a 256^3 GEMM (`--trace_so`). Dependencies are an explicit
dataflow DAG (SRAM last-writer per buffer + the vcix preload/matmul FSM). An
asynchronous DMA is synced to the consumer of its data by the **runtime tag
slot** `(tag_id, tag_slot)` through an explicit `togsim.memory_barrier` (lowered
from the source `memref.dma_wait`); a sync DMA is blocking. ABI is **v11**. An
earlier design used a compile-time `event_id` / heap event handle with
`wait`/`signal`; it was removed because one static DMA op runs once per loop
iteration into a different tag slot, which a compile-time id cannot pair per
iteration. **Validation:** on the 256^3 GEMM with the real gem5 cycle table, the
trace path totals **2518 cycles** vs the legacy path's **2698** through the real
Core; all togsim python tests pass; TOGSim builds.

## Branch

- Work branch: `feature/togsim-cpp-trace` (PR #267 -> develop)

## Status

| Milestone | State |
|---|---|
| P0 — ABI header + op vocabulary | DONE (ABI evolved to v11) |
| P1 — `build_skeleton` pass | DONE, verified — runs on a real GEMM fixture, module verifies, compute grouping + dma/barrier counts match the legacy `build_tog` TOG. |
| P2 — togsim -> emitc -> cpp -> .so | DONE — `lower_to_emitc.py` builds EmitC, `mlir-translate` -> C++, `g++ -shared` -> `.so`; validated by build/symbol checks and a dlopen run harness. |
| P3 — TOGSim loader + runtime + cycle table; real-Core run | DONE — runs end-to-end through the real Simulator/Core (256^3 GEMM, `--trace_so`). Runtime tag-slot pairing (ABI v11, `togsim.memory_barrier`), explicit dataflow DAG (read/write_bufs last-writer + vcix FSM), real tile addresses, cycle_table. `togsim_runtime.cc`/`togsim_loader.h`/`togsim_trace_bridge.cc` feed TraceRec into the real Core. Cycle comparison vs legacy on the real gem5 table: trace 2518 vs legacy 2698. Legacy ONNX-TOG path DEPRECATED in place, kept live. |
| P4 — symbolic-bound dynamic shape, streaming sink | not started |
| P5 — op-family migration (conv/SDPA/vector) | not started |

### Async-DMA sync: runtime tag slot (current), event-id (removed)

The original P1 threaded the dma->wait dependency as an SSA `!togsim.event`
value, which fails `module.verify()` on a software-pipelined kernel (the
`togsim.dma` sits in the prefetch loop nest, its consumer in a sibling compute
nest, so the value does not dominate its use). An intermediate fix used a
compile-time `event_id` attribute (later a heap-allocated event handle). Both
were **removed**: one static `togsim.dma` op executes once per loop iteration
into a *different* runtime tag slot, so a compile-time id (one per static op)
cannot pair iteration i's DMA with iteration i's wait.

Current mechanism (ABI v11): `togsim.dma` carries `tag_id` (its tag-memref
identity) plus the runtime tag-index operand `%tag[%idx]` and returns void. The
source `memref.dma_wait` is mapped through to an explicit
`togsim.memory_barrier {tag_id, write_bufs}` carrying the runtime tag index. At
runtime an async DMA and its barrier are paired by `(tag_id, tag_slot)` through
the existing Core tag table (`prepare_tag_key`/`set_tag_finish`/
`register_tag_waiter`): the DMA signals at data arrival, the barrier waits, and
the barrier becomes the loaded buffer's last-writer so consumers gate on
arrival. (The one remaining auto-inserted barrier is `togsim.compute_barrier`,
the compute fence before a store — marked FIXME to become explicit later.)

### P2 decisions

* **ABI v11 (runtime tag slot).** `togsim_dma` returns void and carries
  `(is_async, tag_id, tag_slot, read_bufs, write_bufs)`. The
  `togsim_memory_barrier(tag_id, tag_slot, write_bufs)` is the explicit
  async-DMA sync. No `event_id`, no event handle, no `wait`/`signal`.
* **C4 drives the upstream EmitC conversion passes** (it does not hand-build
  EmitC). It only does the parts upstream cannot: rewrite the *unregistered*
  `togsim.*` ops to `emitc.call_opaque` and rewrite the kernel signature to the
  ABI form. Then it runs, in-process (`mlir.passmanager`),
  `func.func(lower-affine), convert-scf-to-emitc, convert-arith-to-emitc,
  convert-func-to-emitc`. One local fixup: in this LLVM 20 build
  `convert-scf-to-emitc` emits `emitc.for` with `index` bounds, so
  `convert-arith-to-emitc` (constants -> `!emitc.size_t`) leaves
  `unrealized_conversion_cast` on the bounds that nothing folds and
  `mlir-to-cpp` can't print (design sec 8 risk). `_fold_for_bound_casts`
  rewrites those bound constants to `index`-typed `emitc.constant`, clearing
  the casts. (`emitc.for` *does* accept `size_t` bounds with an explicit
  `: !emitc.size_t`, but keeping the bounds `index` avoids retyping the IV.)
* **Addresses (wired in P3, approach A):** `togsim_dma` passes `(arg_id, element
  offset)` with the offset computed from the loop IVs; the runtime adds the
  tensor base. `togsim.compute` is keyed by `tile_id` for cost.

## Files (key)

- `TOGSim/include/togsim_runtime.h` — extern "C" ABI v11 (`togsim_dma`,
  `togsim_memory_barrier`, `togsim_compute`, `togsim_compute_barrier`,
  `togsim_core_alloc`, `togsim_emit` entry, `TOGSIM_ABI_VERSION`, opaque
  `EmitCtx`).
- `PyTorchSimFrontend/mlir/passes/togsim_ops.py` — single source of truth for the
  skeleton+API MLIR vocabulary (op names, attr keys, op->callee map).
- `PyTorchSimFrontend/mlir/passes/build_skeleton.py` + `dep_analysis.py` — the P1
  pass + dependency analysis (reuse build_tog's `TogBuilder`/`_build`; map
  dma_start->togsim.dma, dma_wait->togsim.memory_barrier, attach read/write_bufs;
  use-based DCE).
- `TOGSim/src/togsim_runtime.cc`, `TOGSim/include/togsim_loader.h`,
  `TOGSim/src/togsim_trace_bridge.cc` — C6 runtime, dlopen loader, and the bridge
  that feeds the recorded TraceRec stream into the real Core.
- `tests/test_togsim_skeleton.py` — `test_togsim_ops_contract` (runs anywhere) +
  `test_build_skeleton_on_fixture` (gated on bindings + a fixture).
- `PyTorchSimFrontend/mlir/passes/lower_to_emitc.py` — the P2/C4 pass: skeleton
  module -> EmitC `togsim_emit` -> C++ (`mlir-translate`) -> `.so` (`g++`).
  Entry points: `lower_to_emitc(module)`, `build_trace_so(postvcix_path, so)`,
  and a `__main__` CLI (`--so`, `--emit-cpp`, `--include-dir`).
- `tests/test_togsim_emitc.py` — `test_build_trace_so` (EmitC + symbol checks) +
  `test_trace_so_runs` (dlopen the `.so` against a stub runtime, run it). Gated
  on bindings + `mlir-translate` + a C++ compiler + the fixture.

## Reproduce P1 + P2 (one GEMM kernel)

```bash
# 1. post-vcix fixture: compile a GEMM (needs the built PyTorchSimDevice .so).
export pytorchsim_functional_mode=False
python tests/ops/gemm/test_matmul.py
FIX=$(find "${TORCHSIM_DUMP_PATH:-.}" -name '*_postvcix.mlir' | head -1)
# build_skeleton/lower_to_emitc only need the .mlir + bindings, not torch, so a
# fixture compiled in any worktree is fine.

# 2. P1: skeleton+API MLIR.
python -m PyTorchSimFrontend.mlir.passes.build_skeleton "$FIX" --out /tmp/skel.mlir
#   stderr: "skeleton: compute=.. dma=.. memory_barrier=.."

# 3. P2: skeleton -> EmitC -> C++ -> .so (reads skel from $FIX via build_skeleton).
python -m PyTorchSimFrontend.mlir.passes.lower_to_emitc "$FIX" \
    --so /tmp/trace.so --emit-cpp /tmp/trace.cpp
nm -D /tmp/trace.so | grep togsim     # togsim_emit = T; togsim_dma/memory_barrier/compute = U

# 4. tests
TOGSIM_SKELETON_FIXTURE="$FIX" python -m pytest \
    tests/test_togsim_skeleton.py tests/test_togsim_emitc.py -q
```

Note: `mlir-opt`/`mlir-translate` live in `$TORCHSIM_LLVM_PATH` but are not on
`$PATH`; `lower_to_emitc` resolves `mlir-translate` from `TORCHSIM_LLVM_PATH`.

## Next steps (P3 is done; remaining work)

The producer is wired into TOGSim and runs through the real Core (trace 2518 vs
legacy 2698 on the 256^3 GEMM). The parallelism / reduction / core-dispatch
design is in `togsim_cpp_trace.md` §9. Summary: the producer is core-transparent
(knows nothing about `num_cores`); it enumerates parallel output-tile work-items
and calls `togsim_core_alloc` at each work-item boundary. Parallel = independent
work-items; reduction = program order inside one work-item; core binding = the
`togsim_core_alloc` runtime callback (policy lives in TOGSim). Async-DMA data
sync = the runtime `(tag_id, tag_slot)` via `togsim.memory_barrier`. `num_cores`
is extrinsic so it is never baked; vlane/tile sizes are intrinsic and stay baked.
Split-K is a deferred exception.

Remaining (priority order; full list in STATUS §7 and design §11.2):

- **SRAM tile lifecycle (double-buffer throttle).** `togsim.dma` carries
  `tag_slot` (the SRAM slot key); the consumer must use it to throttle in-flight
  loads to the buffer depth on multi-tile / double-buffered kernels.
- **Preload concurrency cap (design §10.5).** Give a preload a non-zero occupancy
  (its weight-load time) so concurrent preloads are capped at the SA count.
  Pre-existing in BOTH paths.
- **Per-output-tile dispatch / multi-core.** One `togsim_core_alloc` per
  work-item today; distribute independent output tiles across cores.
- **Robust gem5 cycle_list wiring.** The extension_codecache
  `TORCHSIM_DUMP_TRACE_SO=1` hook is flaky under concurrent compiles.
- **P5 op coverage** (conv/SDPA/vector) and **P4** (symbolic shape, streaming
  sink), then **retire the legacy ONNX-TOG path**.

Full design: `togsim_cpp_trace.md` §5-11.

## Environment requirements (for the new session)

- MLIR Python bindings importable (`import mlir.ir`). They ship with the LLVM
  build at `${TORCHSIM_LLVM_PATH%/bin}/python_packages/mlir_core`; the CI docker
  image `ghcr.io/psal-postech/torchsim-ci` has them. `passes/__init__` also
  derives the path from `TORCHSIM_LLVM_PATH`.
- `pytest` to run the test files directly (`pip install pytest` if absent).
- `mlir-translate` (in `$TORCHSIM_LLVM_PATH`) and a host C++ compiler (`g++`/
  `$CXX`) for the P2 `.so` path.
- TOGSim build (for `--trace_so`): `cd TOGSim/build && cmake ..
  -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`. The Simulator target has
  ENABLE_EXPORTS so a dlopen'd `.so` resolves the `togsim_*` callbacks.
- When iterating on passes, clear the codegen caches (`$TORCHSIM_DUMP_PATH`,
  default `outputs/`) between runs — see CLAUDE.md "Codegen changes are sticky".

## Verification that already passes anywhere (sanity)

```bash
python -m py_compile PyTorchSimFrontend/mlir/passes/build_skeleton.py \
    PyTorchSimFrontend/mlir/passes/togsim_ops.py tests/test_togsim_skeleton.py
# contract test (no bindings needed): see test_togsim_ops_contract
```
