#pragma once
// togsim_runtime.h -- C ABI between a compiled, shape-parametric trace producer
// (`.so`, MLIR -> EmitC -> C++) and TOGSim: each call emits one modeled
// instruction, and the producer carries no timing model.  See sec 5.4 of the doc.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Producer/runtime ABI version. TOGSim refuses to load a producer whose
// embedded togsim_abi_version() does not match TOGSIM_ABI_VERSION.
#define TOGSIM_ABI_VERSION 12
int32_t togsim_abi_version(void);

// Opaque per-invocation context owned by TOGSim. Holds the record sink and the
// tile_id->cycle lookup. Never dereferenced by the producer.
typedef struct EmitCtx EmitCtx;

// Direction for togsim_dma.
typedef enum {
  TOGSIM_DMA_LOAD  = 0,  // DRAM -> SRAM (MOVIN)
  TOGSIM_DMA_STORE = 1,  // SRAM -> DRAM (MOVOUT)
} togsim_dma_dir;

// Emit a DMA (sec 5.4). `offset` is an ELEMENT offset into tensor `arg_id`; null
// `strides` => contiguous. `is_async` => it finishes at ISSUE, so the barrier keyed
// by `(tag_id, tag_slot)` gates consumers. `read_bufs`/`write_bufs` -> sec 10.

void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, int32_t tag_id, uint64_t tag_slot,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write);

// Emit a fixed-size tile compute. Cost comes from the tile_id->cycle table (sec 6),
// not from `dims`. `compute_type` (0 vector / 1 matmul / 2 preload) routes the op to
// the VPU or the systolic array.
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);

// The explicit async-DMA sync (sec 10.5). Pairs with its async togsim_dma by the
// runtime `(tag_id, tag_slot)` and becomes the last writer of `write_bufs`, so
// consumers gate on data arrival. A sync dma blocks to arrival and needs no barrier.
void togsim_memory_barrier(EmitCtx* ctx, int32_t tag_id, uint64_t tag_slot,
                           const int64_t* write_bufs, int32_t n_write);

// A parallel work-item body, outlined by the producer (sec 9.3): `iv` holds the
// packed parallel loop indices (e.g. the (m,n) output-tile indices). One uniform
// signature => one general dispatcher serves every kernel. The runtime only reads iv.
typedef void (*togsim_tile_fn)(EmitCtx* ctx, int64_t* iv, int32_t n_iv);

// Dispatch one work-item (sec 9.3): round-robin a core, bracket `fn` with
// TILE_BEGIN/TILE_END, and invoke it -- so the work-item scope IS the call. Core
// choice is runtime-owned; the producer never names num_cores or a core.
void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn,
                     int64_t* iv, int32_t n_iv);

// Entry point the loader resolves in the producer `.so`. `shape_args` carries
// the runtime values for the kernel's symbolic dimensions (in a kernel-specific
// order recorded alongside the cached `.so`); `n_shape_args` is their count.
void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args);

#ifdef __cplusplus
}  // extern "C"
#endif
