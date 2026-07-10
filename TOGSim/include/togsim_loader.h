#pragma once
// togsim_loader.h -- the TOGSim half (not the producer ABI): `dlopen` a producer
// `.so`, run its `togsim_kernel`, record the emitted instructions.  The
// "materializing sink" of sec 5.3 / 9.7; the stream goes to togsim_trace_bridge.h.

#include <cstdint>
#include <functional>
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

// Streaming variant: feeds each emitted record to `sink` and retains NOTHING.
// The whole recorded stream is O(#tiles) and, for a small systolic array, that is
// millions of records -- materializing it in a RunResult AND then building the
// TileGraph from it means both live at peak (measured: ~equal halves of peak RSS,
// SIGKILL on large 8x8 convs). Callers that need two passes over the stream simply
// run the producer twice; togsim_kernel is a pure emitter, so replaying it is
// cheap and yields an identical stream. Returns ok.
using TraceSink = std::function<void(const TraceRec&)>;
bool run_producer_stream(const char* so_path,
                         const int64_t* shape_args, int32_t n_shape,
                         const uint64_t* tensor_base, int32_t n_tensors,
                         const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                         const int32_t* partition_cores, int32_t n_partition_cores,
                         const TraceSink& sink);
// ---------------------------------------------------------------------------
// On-demand (lazy) production of one work-item at a time.
//
// `togsim_dispatch(ctx, fn, iv, n)` hands us the work-item's function pointer
// and its parallel induction variables, and the tile body reads nothing but
// `ctx` and `iv`. So a single indexing pass -- which records (fn, iv, core) and
// SKIPS the call -- is enough to invoke any work-item later, on its own, with no
// replay of the others. That makes the TileGraph buildable one dispatch tile at
// a time, single-threaded: peak memory is O(tiles in flight), not O(dispatches).
//
// Legal because a dispatch tile is dependency-closed: the bridge resets its
// writers/seeds/tag maps at every tile boundary, so no dependency edge crosses
// tiles (measured: cross_tile_edges == 0).
struct WorkItem {
  void* fn = nullptr;           // togsim_tile_fn
  std::vector<int64_t> iv;      // the enclosing parallel loop indices
  int32_t core = 0;             // round-robin binding, fixed at index time
};

class LazyProducer {
 public:
  LazyProducer() = default;
  ~LazyProducer();
  LazyProducer(const LazyProducer&) = delete;
  LazyProducer& operator=(const LazyProducer&) = delete;

  // dlopen the .so and run togsim_kernel once, in INDEX mode: every
  // togsim_dispatch is recorded (so it can be re-invoked alone later) AND run,
  // with every record streamed to `sink`. That single run doubles as the
  // footprint pre-pass, so records emitted outside a dispatch are not lost.
  bool open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
            const uint64_t* tensor_base, int32_t n_tensors,
            const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
            const int32_t* partition_cores, int32_t n_partition_cores,
            const TraceSink* sink = nullptr);

  size_t num_items() const { return _items.size(); }
  // Emit work-item `i`'s record stream (TILE_BEGIN, body, TILE_END) into `sink`.
  void run_item(size_t i, const TraceSink& sink);
  // Records the producer emitted OUTSIDE any dispatch. The TileGraph builder
  // drops these (they belong to no work-item) exactly as the eager builder did;
  // they still reach the footprint pre-pass through open()'s sink.
  uint64_t stray_records() const { return _stray; }

 private:
  struct EmitCtx* _ctx = nullptr;   // opaque; owned
  void* _lib = nullptr;
  std::vector<WorkItem> _items;
  uint64_t _stray = 0;
  std::vector<uint64_t> _bases;
  std::vector<int64_t> _cyc, _ovl;
};

}  // namespace togsim
