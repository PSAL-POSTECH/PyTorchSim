#pragma once
// togsim_runtime.h -- C ABI between a compiled, shape-parametric trace producer
// (`.so`, MLIR -> EmitC -> C++) and TOGSim: each call emits one modeled
// instruction, and the producer carries no timing model.  See sec 5.4 of the doc.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Producer/runtime ABI version. A producer TU including this header defines the weak
// togsim_producer_abi_version it was built against (TOGSim's TUs, TOGSIM_HOST, only
// declare it); LazyProducer::open refuses a producer that lacks it or differs.
#define TOGSIM_ABI_VERSION 14
#ifdef TOGSIM_HOST
extern const int32_t togsim_producer_abi_version;
#else
__attribute__((weak, used, visibility("default")))
extern const int32_t togsim_producer_abi_version = TOGSIM_ABI_VERSION;
#endif

// Opaque per-invocation context owned by TOGSim. Holds the recorded trace and
// the tile_id->cycle lookup. Never dereferenced by the producer.
typedef struct EmitCtx EmitCtx;

// Direction for togsim_dma.
typedef enum {
  TOGSIM_DMA_LOAD  = 0,  // DRAM -> SRAM (MOVIN)
  TOGSIM_DMA_STORE = 1,  // SRAM -> DRAM (MOVOUT)
} togsim_dma_dir;

// Emit a DMA (sec 5.4). `offset` is an ELEMENT offset into tensor `arg_id`; null
// `strides` => contiguous. `is_async` => it finishes at ISSUE and signals semaphore
// `sem` with its element count; the barrier on `sem` gates consumers. `read_bufs`/
// `write_bufs` -> sec 10. A store's `sem` is ignored (its waits are not lowered).
// `indirect` => each element also moves by what its index added in the functional
// run: dump indirect_index_<key>_<n>.raw, key = the work-item's togsim_dispatch key,
// n = this dma's rank among the work-item's indirect dmas.

// --- BEGIN trace-producer call formats (copied verbatim into generated trace.cpp) ---
// Each togsim_* call below lowers 1:1 to one of these free functions. Arg formats:
//   togsim_dma(ctx, dir, arg_id, offset, ndim, dims[], strides[], elem_bits,
//              is_async, sem, read_bufs[], n_read, write_bufs[], n_write, indirect)
//              dir: 0=load (MOVIN), 1=store (MOVOUT); indirect: 0 or 1
//   togsim_compute(ctx, tile_id, compute_type, ndim, dims[], read_bufs[], n_read,
//                  write_bufs[], n_write)   compute_type: 0=vector, 1=matmul, 2=preload,
//                                          3=cross-lane
//   togsim_memory_barrier(ctx, sem, expected)   // expected = elements signaled on sem
//   togsim_dispatch(ctx, tile_fn, iv[], n_iv, key)   // run one work-item
//   togsim_kernel(ctx, shape_args[], n_shape_args)   // producer entry point
// --- END trace-producer call formats ---
void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, uint64_t sem,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write,
                int32_t indirect);

// Emit a fixed-size tile compute. Cost comes from the tile_id->cycle table (sec 6),
// not from `dims`. `compute_type` (0 vector / 1 matmul / 2 preload / 3 cross-lane) routes
// the op to the VPU, the systolic array or the cross-lane unit.
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);

// The explicit async-DMA sync (sec 10.5): waits every async load signaled on `sem`
// since its last wait, whose element counts must sum to `expected`, and becomes the
// last writer of their buffers. A sync dma blocks to arrival and needs no barrier.
void togsim_memory_barrier(EmitCtx* ctx, uint64_t sem, int64_t expected);

// A parallel work-item body, outlined by the producer (sec 9.3): `iv` holds the
// packed parallel loop indices (e.g. the (m,n) output-tile indices). One uniform
// signature => one general dispatcher serves every kernel. The runtime only reads iv.
typedef void (*togsim_tile_fn)(EmitCtx* ctx, int64_t* iv, int32_t n_iv);

// Dispatch one work-item (sec 9.3): round-robin a core, bracket `fn` with
// TILE_BEGIN/TILE_END, and invoke it -- so the work-item scope IS the call. Core
// choice is runtime-owned; the producer never names num_cores or a core. `key` is
// the work-item's dma_index_key in the functional run, naming its index dumps.
void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn,
                     int64_t* iv, int32_t n_iv, int64_t key);

// Entry point the loader resolves in the producer `.so`. `shape_args` carries
// the runtime values for the kernel's symbolic dimensions (in a kernel-specific
// order recorded alongside the cached `.so`); `n_shape_args` is their count.
void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args);

#ifdef __cplusplus
}  // extern "C"
#endif
