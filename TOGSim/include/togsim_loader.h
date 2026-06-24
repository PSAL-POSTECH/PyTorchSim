#pragma once
// togsim_loader.h
// -----------------------------------------------------------------------------
// TOGSim-side loader for the compiled trace producer (C6, P3 task 5). NOT part
// of the producer ABI (togsim_runtime.h) -- this is the TOGSim half that
// `dlopen`s a producer `.so`, runs its `togsim_kernel`, and records the emitted
// instruction stream. See docs/design/togsim_cpp_trace.md sec 5.3 / 9.7.
//
// This first cut is the "materializing sink": the callbacks resolve each tile's
// DRAM address (base[arg_id] + offset*elem_bytes) and per-tile compute cost
// (the cycle table), mint event handles, and append a TraceRec per modeled
// instruction. Feeding the recorded stream into the existing timing core
// (Core/Simulator) for cycle-equivalence vs the build_tog path is the remaining
// task-5 step.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "togsim_runtime.h"

namespace togsim {

// One modeled instruction recorded by the runtime callbacks.
struct TraceRec {
  enum Kind { TILE_BEGIN, TILE_END, DMA, COMPUTE, MEMORY_BAR, COMPUTE_BAR } kind;
  int32_t  core;          // work-item -> core binding (set by togsim_dispatch)
  // DMA / MEMORY_BAR
  int32_t  dir;           // togsim_dma_dir
  int32_t  arg_id;        // tensor
  int32_t  elem_bits;
  int32_t  is_async;
  uint64_t addr;          // resolved DRAM byte address = base[arg_id] + off*bytes
  int32_t  tag_id;        // DMA/MEMORY_BAR: tag memref identity; with tag_slot the
                          // runtime pairing key (an async dma <-> its memory_barrier)
  uint64_t tag_slot;      // SRAM tile slot (double-buffer / capacity model)
  std::vector<int64_t> dims;     // tile extents (DMA)
  std::vector<int64_t> strides;  // tile strides (DMA)
  std::vector<int64_t> read_bufs;   // SRAM buffer ids read  (sec 10 dataflow DAG)
  std::vector<int64_t> write_bufs;  // SRAM buffer ids written (MEMORY_BAR: released bufs)
  // COMPUTE
  uint64_t tile_id;
  int32_t  compute_type;  // 0 vector / 1 matmul / 2 preload (Core unit enum)
  int64_t  cycle;         // looked up from the cycle table
  int64_t  overlapping;   // looked up from the cycle table
};

struct RunResult {
  bool ok = false;
  std::vector<TraceRec> trace;
};

// Load `so_path`, run its `togsim_kernel(shape_args, n_shape)` against a freshly
// built EmitCtx, and return the recorded trace.
//   tensor_base[arg_id] : DRAM base address of each kernel tensor argument
//   cyc[tile_id] / ovl[tile_id] : the cycle table (cycle, overlapping_cycle)
//   num_cores : dispatch round-robins work-items across this many cores
RunResult run_producer(const char* so_path,
                       const int64_t* shape_args, int32_t n_shape,
                       const uint64_t* tensor_base, int32_t n_tensors,
                       const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                       int32_t num_cores);

// First-order reference timing over a recorded trace, to validate that the
// stream carries enough to be scheduled (it is NOT the production Core -- no
// DRAM/NoC/L2 contention; the real cycle-equivalence path feeds Tile/TileGraph
// into Core). Models, per core: a DMA-engine timeline (DMAs serialize, overlap
// compute), a compute timeline (serial = reduction accumulate, with the
// finish = prev.finish + cycle - overlapped pipeline overlap of Core.cc), and
// data dependencies (a compute waits the dmas whose handles its preceding
// togsim_wait()s named).
struct TimingParams { uint64_t dma_latency = 100; };
struct SimResult { uint64_t total_cycle = 0; int n_compute = 0, n_dma = 0; };
SimResult simulate(const RunResult& run, const TimingParams& params);

}  // namespace togsim
