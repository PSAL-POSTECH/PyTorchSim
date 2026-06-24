#pragma once
// togsim_loader.h -- the TOGSim half (not the producer ABI): `dlopen` a producer
// `.so`, run its `togsim_kernel`, record the emitted instructions.  The
// "materializing sink" of sec 5.3 / 9.7; the stream goes to togsim_trace_bridge.h.

#include <cstdint>
#include <vector>

#include "togsim_runtime.h"

namespace togsim {

// One modeled instruction recorded by the runtime callbacks.
struct TraceRec {
  enum Kind { TILE_BEGIN, TILE_END, DMA, COMPUTE, MEMORY_BAR } kind;
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
  std::vector<int64_t> read_bufs;   // SRAM buffer ids read  (sec 10 dependency model)
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

// Load `so_path`, run its `togsim_kernel`, and return the recorded trace.
// `tensor_base` gives each tensor argument's DRAM base, `cyc`/`ovl` the cycle table.
// Work-items round-robin only over `partition_cores` (empty/null -> core 0).
RunResult run_producer(const char* so_path,
                       const int64_t* shape_args, int32_t n_shape,
                       const uint64_t* tensor_base, int32_t n_tensors,
                       const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                       const int32_t* partition_cores, int32_t n_partition_cores);

}  // namespace togsim
