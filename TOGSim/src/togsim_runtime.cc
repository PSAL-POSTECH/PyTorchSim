// togsim_runtime.cc -- the producer ABI (togsim_runtime.h) and the loader
// (togsim_loader.h). The producer's calls each record a TraceRec on the opaque
// EmitCtx, resolving DRAM addresses and per-tile cycles as they go.

#include "togsim_loader.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <utility>
#include <vector>

// Full definition of the opaque handle from togsim_runtime.h. The producer holds
// only EmitCtx* and never dereferences it.
struct EmitCtx {
  // inputs supplied by the loader
  const uint64_t* tensor_base = nullptr;
  int32_t         n_tensors = 0;
  const int64_t*  cyc = nullptr;   // tile_id -> cycle
  const int64_t*  ovl = nullptr;   // tile_id -> overlapping_cycle
  int32_t         n_tiles = 0;
  std::vector<int32_t> cores{0};   // the partition's core ids; dispatch round-robins over these
  // mutable run state
  int32_t  rr = 0;            // round-robin cursor into `cores`
  int32_t  cur_core = -1;     // current work-item's core
  // INDEX mode (LazyProducer::open): togsim_dispatch additionally records the
  // work-item so it can be re-invoked on its own later. Records emitted outside
  // any dispatch (depth == 0) are counted; the TileGraph builder drops them, so
  // only the footprint pre-pass ever sees them.
  bool                  capture = false;
  std::vector<togsim::WorkItem>* items = nullptr;
  int                   depth = 0;
  uint64_t              stray = 0;
  // Scratch record reused by every callback. The sink consumes a record before
  // the next one is emitted (and run_producer copies it), so one buffer is
  // enough -- and its vectors keep their capacity, which is the difference
  // between four allocations per emitted record and none.
  togsim::TraceRec scratch;
  // Exactly one of these is active. With a sink the record is consumed and dropped
  // (streaming, O(1) memory); without one it is appended (legacy run_producer).
  const togsim::TraceSink* sink = nullptr;
  std::vector<togsim::TraceRec> trace;
};

namespace {
// Reset `ctx->scratch` for a new record. clear() keeps each vector's capacity.
inline togsim::TraceRec& blank(EmitCtx* ctx, togsim::TraceRec::Kind k, int32_t core) {
  togsim::TraceRec& r = ctx->scratch;
  r.dims.clear(); r.strides.clear(); r.read_bufs.clear(); r.write_bufs.clear();
  r.kind = k; r.core = core;
  r.dir = 0; r.arg_id = 0; r.elem_bits = 0; r.is_async = 0;
  r.addr = 0; r.tag_id = 0; r.tag_slot = 0;
  r.tile_id = 0; r.compute_type = 0; r.cycle = 0; r.overlapping = 0;
  return r;
}
inline void emit_rec(EmitCtx* ctx, const togsim::TraceRec& r) {
  if (ctx->depth == 0) ++ctx->stray;            // emitted outside any dispatch
  if (ctx->sink) (*ctx->sink)(r);
  else ctx->trace.push_back(r);
}
}  // namespace

extern "C" {

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn, int64_t* iv, int32_t n_iv) {
  // Work-item wrapper (sec 9.3): round-robin over THIS partition's cores only --
  // a work-item on another partition's core would sit in this partition's scheduler
  // forever. TILE_BEGIN/TILE_END bracket the ops `fn` emits under ctx->cur_core.
  ctx->cur_core = ctx->cores.empty() ? 0
                : ctx->cores[ctx->rr++ % (int32_t)ctx->cores.size()];
  if (ctx->capture) {   // index pass: remember the work-item so it can be re-run alone
    togsim::WorkItem w;
    w.fn = (void*)fn;
    w.core = ctx->cur_core;
    if (iv && n_iv > 0) w.iv.assign(iv, iv + n_iv);
    ctx->items->push_back(std::move(w));
  }
  ++ctx->depth;
  emit_rec(ctx, blank(ctx, togsim::TraceRec::TILE_BEGIN, ctx->cur_core));
  fn(ctx, iv, n_iv);
  emit_rec(ctx, blank(ctx, togsim::TraceRec::TILE_END, ctx->cur_core));
  --ctx->depth;
}

void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, int32_t tag_id, uint64_t tag_slot,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write) {
  uint64_t base = (arg_id >= 0 && arg_id < ctx->n_tensors)
                      ? ctx->tensor_base[arg_id] : 0;
  uint64_t addr = base + offset * (uint64_t)(elem_bits / 8);
  togsim::TraceRec& r = blank(ctx, togsim::TraceRec::DMA, ctx->cur_core);
  r.dir = dir; r.arg_id = arg_id; r.elem_bits = elem_bits;
  r.is_async = is_async; r.addr = addr; r.tag_id = tag_id; r.tag_slot = tag_slot;
  if (dims) r.dims.reserve(ndim);
  if (strides) r.strides.reserve(ndim);
  r.read_bufs.reserve(n_read);
  r.write_bufs.reserve(n_write);
  for (int32_t i = 0; i < ndim; ++i) {
    if (dims) r.dims.push_back(dims[i]);
    if (strides) r.strides.push_back(strides[i]);
  }
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  emit_rec(ctx, r);
}

void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write) {
  (void)ndim; (void)dims;
  togsim::TraceRec& r = blank(ctx, togsim::TraceRec::COMPUTE, ctx->cur_core);
  r.tile_id = tile_id;
  r.compute_type = compute_type;
  r.read_bufs.reserve(n_read);
  r.write_bufs.reserve(n_write);
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  if (ctx->cyc && (int32_t)tile_id < ctx->n_tiles) r.cycle = ctx->cyc[tile_id];
  if (ctx->ovl && (int32_t)tile_id < ctx->n_tiles) r.overlapping = ctx->ovl[tile_id];
  emit_rec(ctx, r);
}

void togsim_memory_barrier(EmitCtx* ctx, int32_t tag_id, uint64_t tag_slot,
                           const int64_t* write_bufs, int32_t n_write) {
  togsim::TraceRec& r = blank(ctx, togsim::TraceRec::MEMORY_BAR, ctx->cur_core);
  r.tag_id = tag_id; r.tag_slot = tag_slot;
  r.write_bufs.reserve(n_write);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  emit_rec(ctx, r);
}

}  // extern "C"

namespace togsim {

namespace {
void init_ctx(EmitCtx& ctx, const uint64_t* tensor_base, int32_t n_tensors,
              const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
              const int32_t* partition_cores, int32_t n_partition_cores) {
  ctx.tensor_base = tensor_base; ctx.n_tensors = n_tensors;
  ctx.cyc = cyc; ctx.ovl = ovl; ctx.n_tiles = n_tiles;
  ctx.cores.assign(partition_cores, partition_cores + (n_partition_cores > 0 ? n_partition_cores : 0));
  if (ctx.cores.empty()) ctx.cores.push_back(0);
}
// dlopen the producer and run its togsim_kernel against `ctx`.
bool load_and_run(const char* so_path, const int64_t* shape_args, int32_t n_shape,
                  EmitCtx& ctx) {
  void* lib = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!lib) { fprintf(stderr, "togsim: dlopen failed: %s\n", dlerror()); return false; }
  auto emit = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(lib, "togsim_kernel");
  if (!emit) { fprintf(stderr, "togsim: dlsym togsim_kernel failed: %s\n", dlerror()); return false; }
  emit(&ctx, (int64_t*)shape_args, n_shape);
  return true;
}
}  // namespace

RunResult run_producer(const char* so_path,
                       const int64_t* shape_args, int32_t n_shape,
                       const uint64_t* tensor_base, int32_t n_tensors,
                       const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                       const int32_t* partition_cores, int32_t n_partition_cores) {
  RunResult res;
  EmitCtx ctx;
  init_ctx(ctx, tensor_base, n_tensors, cyc, ovl, n_tiles, partition_cores, n_partition_cores);
  if (!load_and_run(so_path, shape_args, n_shape, ctx)) return res;
  res.ok = true;
  res.trace = std::move(ctx.trace);
  return res;
}

LazyProducer::~LazyProducer() {
  delete _ctx;   // _lib intentionally left open: the tile fns must stay callable
}

bool LazyProducer::open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
                        const uint64_t* tensor_base, int32_t n_tensors,
                        const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                        const int32_t* partition_cores, int32_t n_partition_cores,
                        const TraceSink* sink) {
  // own the tables: the caller's buffers are its locals, and the tile fns are
  // invoked long after it returns.
  if (tensor_base && n_tensors > 0) _bases.assign(tensor_base, tensor_base + n_tensors);
  if (cyc && n_tiles > 0) _cyc.assign(cyc, cyc + n_tiles);
  if (ovl && n_tiles > 0) _ovl.assign(ovl, ovl + n_tiles);

  _ctx = new EmitCtx();
  init_ctx(*_ctx, _bases.empty() ? nullptr : _bases.data(), (int32_t)_bases.size(),
           _cyc.empty() ? nullptr : _cyc.data(), _ovl.empty() ? nullptr : _ovl.data(),
           (int32_t)_cyc.size(), partition_cores, n_partition_cores);

  _lib = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!_lib) { fprintf(stderr, "togsim: dlopen failed: %s\n", dlerror()); return false; }
  auto kernel = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(_lib, "togsim_kernel");
  if (!kernel) { fprintf(stderr, "togsim: dlsym togsim_kernel failed: %s\n", dlerror()); return false; }

  // One full producer run: it both indexes the dispatches and streams every
  // record (including any emitted outside a dispatch) to `sink`. This is the
  // footprint pre-pass the eager builder used to run, so nothing is missed.
  _ctx->capture = true;
  _ctx->items = &_items;
  _ctx->sink = sink;
  kernel(_ctx, (int64_t*)shape_args, n_shape);
  _ctx->capture = false;
  _ctx->items = nullptr;
  _ctx->sink = nullptr;
  _stray = _ctx->stray;
  return true;
}

void LazyProducer::run_item(size_t i, const TraceSink& sink) {
  if (i >= _items.size()) return;
  WorkItem& w = _items[i];
  _ctx->sink = &sink;
  _ctx->cur_core = w.core;                     // the binding fixed at index time
  _ctx->depth = 1;
  emit_rec(_ctx, blank(_ctx, TraceRec::TILE_BEGIN, w.core));
  ((togsim_tile_fn)w.fn)(_ctx, w.iv.empty() ? nullptr : w.iv.data(), (int32_t)w.iv.size());
  emit_rec(_ctx, blank(_ctx, TraceRec::TILE_END, w.core));
  _ctx->depth = 0;
  _ctx->sink = nullptr;
}
}  // namespace togsim
