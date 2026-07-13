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
  int32_t  cur_core = -1;     // core of the work-item being replayed
  std::vector<togsim::WorkItem> items;   // every dispatch, recorded by togsim_kernel
  std::vector<togsim::TraceRec> trace;   // records of the work-item being replayed
};

namespace {
// A fresh TraceRec of kind `k` for core `core` (fields zeroed).
inline togsim::TraceRec blank(togsim::TraceRec::Kind k, int32_t core) {
  togsim::TraceRec r{};
  r.kind = k;
  r.core = core;
  return r;
}
inline void emit_rec(EmitCtx* ctx, const togsim::TraceRec& r) {
  ctx->trace.push_back(r);
}
}  // namespace

extern "C" {

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn, int64_t* iv, int32_t n_iv) {
  // Register the work-item; LazyProducer::run_item runs its body later, on demand.
  // Round-robin over THIS partition's cores only -- a work-item on another
  // partition's core would sit in this partition's scheduler forever.
  togsim::WorkItem w;
  w.fn = (void*)fn;
  w.core = ctx->cores.empty() ? 0 : ctx->cores[ctx->rr++ % (int32_t)ctx->cores.size()];
  if (iv && n_iv > 0) w.iv.assign(iv, iv + n_iv);
  ctx->items.push_back(std::move(w));
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
  togsim::TraceRec r = blank(togsim::TraceRec::DMA, ctx->cur_core);
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
  togsim::TraceRec r = blank(togsim::TraceRec::COMPUTE, ctx->cur_core);
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
  togsim::TraceRec r = blank(togsim::TraceRec::MEMORY_BAR, ctx->cur_core);
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
}  // namespace

LazyProducer::~LazyProducer() {
  delete _ctx;   // _lib intentionally left open: the tile fns must stay callable
}

bool LazyProducer::open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
                        const uint64_t* tensor_base, int32_t n_tensors,
                        const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                        const int32_t* partition_cores, int32_t n_partition_cores) {
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

  // Running the kernel only walks its outer loop nest: every togsim_dispatch
  // registers a work-item and returns, so no tile body runs and no record is
  // emitted yet. run_item() replays an individual work-item later.
  kernel(_ctx, (int64_t*)shape_args, n_shape);
  return true;
}

size_t LazyProducer::num_items() const { return _ctx->items.size(); }

const std::vector<TraceRec>& LazyProducer::run_item(size_t i) {
  _ctx->trace.clear();
  if (i >= _ctx->items.size()) return _ctx->trace;
  WorkItem& w = _ctx->items[i];
  _ctx->cur_core = w.core;                     // the binding fixed when registered
  emit_rec(_ctx, blank(TraceRec::TILE_BEGIN, w.core));
  ((togsim_tile_fn)w.fn)(_ctx, w.iv.empty() ? nullptr : w.iv.data(), (int32_t)w.iv.size());
  emit_rec(_ctx, blank(TraceRec::TILE_END, w.core));
  return _ctx->trace;   // one work-item's records; overwritten by the next call
}
}  // namespace togsim
