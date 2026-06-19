// togsim_runtime.cc
// -----------------------------------------------------------------------------
// C6 runtime + loader for the compiled trace producer (P3 task 5). Implements
// the producer ABI (togsim_runtime.h) and the TOGSim-side loader
// (togsim_loader.h). See docs/design/togsim_cpp_trace.md sec 5.3 / 9.6.1 / 9.7.
//
// The producer `.so` calls the extern "C" togsim_* functions below; each one
// records a TraceRec on the EmitCtx. EmitCtx is the opaque type the producer
// only ever passes back to us. This is the "materializing sink": it resolves
// addresses and per-tile cycles into a recorded instruction stream. Wiring the
// stream into the existing timing core (Core/Simulator) is the remaining step.
// -----------------------------------------------------------------------------

#include "togsim_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <unordered_map>
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
  int32_t         num_cores = 1;
  // mutable run state
  int32_t  rr = 0;            // round-robin core cursor
  int32_t  cur_core = -1;     // current work-item's core
  std::vector<togsim::TraceRec> trace;
};

namespace {
inline togsim::TraceRec blank(togsim::TraceRec::Kind k, int32_t core) {
  togsim::TraceRec r{};
  r.kind = k;
  r.core = core;
  return r;
}
}  // namespace

extern "C" {

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

int32_t togsim_core_alloc(EmitCtx* ctx) {
  // Round-robin a core from the pool; the producer never sees num_cores. Binds
  // it as the current core for the ops that follow (the work-item's reduction).
  ctx->cur_core = ctx->num_cores > 0 ? (ctx->rr++ % ctx->num_cores) : 0;
  ctx->trace.push_back(blank(togsim::TraceRec::DISPATCH, ctx->cur_core));
  return ctx->cur_core;
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
  for (int32_t i = 0; i < ndim; ++i) {
    if (dims) r.dims.push_back(dims[i]);
    if (strides) r.strides.push_back(strides[i]);
  }
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  ctx->trace.push_back(r);
}

void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write) {
  (void)ndim; (void)dims;
  togsim::TraceRec r = blank(togsim::TraceRec::COMPUTE, ctx->cur_core);
  r.tile_id = tile_id;
  r.compute_type = compute_type;
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  if (ctx->cyc && (int32_t)tile_id < ctx->n_tiles) r.cycle = ctx->cyc[tile_id];
  if (ctx->ovl && (int32_t)tile_id < ctx->n_tiles) r.overlapping = ctx->ovl[tile_id];
  ctx->trace.push_back(r);
}

void togsim_memory_barrier(EmitCtx* ctx, int32_t tag_id, uint64_t tag_slot,
                           const int64_t* write_bufs, int32_t n_write) {
  togsim::TraceRec r = blank(togsim::TraceRec::MEMORY_BAR, ctx->cur_core);
  r.tag_id = tag_id; r.tag_slot = tag_slot;
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  ctx->trace.push_back(r);
}

void togsim_compute_barrier(EmitCtx* ctx) {
  ctx->trace.push_back(blank(togsim::TraceRec::COMPUTE_BAR, ctx->cur_core));
}

}  // extern "C"

namespace togsim {

RunResult run_producer(const char* so_path,
                       const int64_t* shape_args, int32_t n_shape,
                       const uint64_t* tensor_base, int32_t n_tensors,
                       const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                       int32_t num_cores) {
  RunResult res;
  void* lib = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!lib) { fprintf(stderr, "togsim: dlopen failed: %s\n", dlerror()); return res; }
  auto emit = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(lib, "togsim_emit");
  if (!emit) { fprintf(stderr, "togsim: dlsym togsim_emit failed: %s\n", dlerror()); return res; }

  EmitCtx ctx;
  ctx.tensor_base = tensor_base; ctx.n_tensors = n_tensors;
  ctx.cyc = cyc; ctx.ovl = ovl; ctx.n_tiles = n_tiles;
  ctx.num_cores = num_cores > 0 ? num_cores : 1;
  emit(&ctx, (int64_t*)shape_args, n_shape);

  res.ok = true;
  res.trace = std::move(ctx.trace);
  return res;
}

SimResult simulate(const RunResult& run, const TimingParams& params) {
  SimResult out;
  std::unordered_map<int, uint64_t> dma_free;     // DMA-engine free time, per core
  std::unordered_map<int, uint64_t> comp_free;    // compute free time, per core
  std::unordered_map<int, uint64_t> prev_comp;    // prev compute finish (overlap), per core
  std::map<std::pair<int32_t, uint64_t>, uint64_t> tag_finish;  // (tag_id,tag_slot) -> finish
  std::vector<uint64_t> pending;                    // barrier-resolved deps since last compute

  for (const auto& t : run.trace) {
    const int c = t.core;
    switch (t.kind) {
      case TraceRec::DMA: {
        // DMAs serialize on the core's DMA engine (overlap compute -> separate
        // timeline). finish = issue + latency, recorded under the runtime tag.
        uint64_t start = dma_free[c];
        uint64_t fin = start + params.dma_latency;
        dma_free[c] = fin;
        tag_finish[{t.tag_id, t.tag_slot}] = fin;
        out.n_dma++;
        break;
      }
      case TraceRec::MEMORY_BAR: {
        // the explicit async-DMA sync: gate the next compute on the paired dma's
        // data-arrival, found by the runtime tag (tag_id, tag_slot).
        auto it = tag_finish.find({t.tag_id, t.tag_slot});
        if (it != tag_finish.end()) pending.push_back(it->second);
        break;
      }
      case TraceRec::COMPUTE: {
        uint64_t deps = 0;
        for (uint64_t f : pending) deps = std::max(deps, f);
        pending.clear();
        uint64_t start = std::max(comp_free[c], deps);
        uint64_t fin;
        auto pit = prev_comp.find(c);
        if (pit != prev_comp.end()) {
          uint64_t prev = pit->second;
          uint64_t tail = prev > start ? prev - start : 0;     // prev still running
          uint64_t overlapped = std::min<uint64_t>(tail, (uint64_t)t.overlapping);
          fin = std::max(start, prev) + (uint64_t)t.cycle - overlapped;
        } else {
          fin = start + (uint64_t)t.cycle;
        }
        comp_free[c] = fin;
        prev_comp[c] = fin;
        out.n_compute++;
        break;
      }
      case TraceRec::DISPATCH:
      case TraceRec::COMPUTE_BAR:
        break;  // work-item boundary / compute fence: no cost in this reference timer
    }
  }
  for (auto& kv : dma_free) out.total_cycle = std::max(out.total_cycle, kv.second);
  for (auto& kv : comp_free) out.total_cycle = std::max(out.total_cycle, kv.second);
  return out;
}

}  // namespace togsim
