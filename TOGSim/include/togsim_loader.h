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
  uint64_t sem;           // DMA/MEMORY_BAR: the semaphore an async load signals / a bar waits
  int64_t  expected;      // MEMORY_BAR: elements the waited batch must have moved
  std::vector<int64_t> dims;     // tile extents (DMA)
  std::vector<int64_t> strides;  // tile strides (DMA)
  std::vector<int64_t> read_bufs;   // SRAM buffer ids read  (sec 10 dependency model)
  std::vector<int64_t> write_bufs;  // SRAM buffer ids written
  int32_t  indirect;      // DMA: its elements also move by the functional run's indices
  int64_t  index_key;     // DMA, indirect: the work-item's key ...
  int64_t  index_seq;     // ... and this dma's rank among its indirect dmas
  // COMPUTE
  uint64_t tile_id;
  int32_t  compute_type;  // 0 vector / 1 matmul / 2 preload / 3 cross-lane (Core unit enum)
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
  int64_t key = 0;              // names the work-item's index dumps (togsim_dispatch)
};

class LazyProducer {
 public:
  LazyProducer() = default;
  ~LazyProducer();
  LazyProducer(const LazyProducer&) = delete;
  LazyProducer& operator=(const LazyProducer&) = delete;

  // dlopen the .so (refused unless its togsim_producer_abi_version is TOGSIM_ABI_VERSION and
  // it states its spad buffer sizes) and run togsim_kernel once: each togsim_dispatch only registers its work-item, so
  // num_items() is known before any record is emitted.
  bool open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
            const uint64_t* tensor_base, int32_t n_tensors,
            const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
            const int32_t* partition_cores, int32_t n_partition_cores);

  size_t num_items() const;
  // The producer's togsim_spad_buffer_bytes, indexed by buffer id (0: not a spad buffer).
  const std::vector<int64_t>& spad_buffer_bytes() const { return _spad_bytes; }
  int64_t item_key(size_t i) const;   // work-item i's togsim_dispatch key
  // Replay work-item `i` and return its record stream (TILE_BEGIN, body,
  // TILE_END). The returned vector is a buffer reused by the next call.
  const std::vector<TraceRec>& run_item(size_t i);

 private:
  struct EmitCtx* _ctx = nullptr;   // opaque; owns the work-item list
  void* _lib = nullptr;
  std::vector<uint64_t> _bases;
  std::vector<int64_t> _cyc, _ovl;
  std::vector<int64_t> _spad_bytes;
};

}  // namespace togsim
