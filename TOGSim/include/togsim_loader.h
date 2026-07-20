#pragma once
// togsim_loader.h -- the TOGSim half (not the producer ABI): `dlopen` a producer
// `.so`, run its `togsim_kernel`, and record the emitted instructions as
// TraceRecs (sec 5.3 / 9.7). togsim_trace_bridge.h turns them into a TileGraph.

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

// The producer is run ON DEMAND, one work-item at a time (materializing the whole
// record stream cost 12.7 GiB on a large 8x8 conv). A tile body reads only ctx and
// iv, so a work-item is (fn, iv, core) -- replayable on its own, whenever needed.
struct WorkItem {
  void* fn = nullptr;           // togsim_tile_fn
  std::vector<int64_t> iv;      // the enclosing parallel loop indices
  int32_t core = 0;             // round-robin binding, fixed when it is registered
};

class LazyProducer {
 public:
  LazyProducer() = default;
  ~LazyProducer();
  LazyProducer(const LazyProducer&) = delete;
  LazyProducer& operator=(const LazyProducer&) = delete;

  // dlopen the .so and run togsim_kernel once: every togsim_dispatch registers
  // its work-item and returns without running the tile body, so num_items() is
  // known but no record is emitted yet.
  bool open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
            const uint64_t* tensor_base, int32_t n_tensors,
            const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
            const int32_t* partition_cores, int32_t n_partition_cores);

  size_t num_items() const;
  // Replay work-item `i` and return its record stream (TILE_BEGIN, body,
  // TILE_END). The returned vector is a buffer reused by the next call.
  const std::vector<TraceRec>& run_item(size_t i);

 private:
  struct EmitCtx* _ctx = nullptr;   // opaque; owns the work-item list
  void* _lib = nullptr;
  std::vector<uint64_t> _bases;
  std::vector<int64_t> _cyc, _ovl;
};

}  // namespace togsim
