#pragma once
// togsim_runtime.h
// -----------------------------------------------------------------------------
// Shared C ABI between a compiled, shape-parametric trace producer (`.so`,
// generated MLIR -> EmitC -> C++) and TOGSim. See docs/design/togsim_cpp_trace.md.
//
// The producer keeps loops as native loops (symbolic bounds become function
// parameters) and calls the functions below; each call emits one trace record =
// one modeled instruction. TOGSim `dlopen`s the producer, constructs an
// `EmitCtx`, calls the entry point, records the emitted stream, and feeds it to
// the existing timing core. The producer carries NO timing model and NO
// functional compute -- it is a deterministic trace generator only.
//
// ABI shape rationale: `mlir-translate --mlir-to-cpp` lowers our `togsim.*` ops
// (via `emitc.call_opaque`) to *free function* calls, so the contract is a set
// of `extern "C"` free functions taking an opaque `EmitCtx*` as the first
// argument. Implementations live in TOGSim and may dispatch internally; the
// `EmitCtx` is opaque to the producer. `togsim_abi_version()` guards against a
// producer `.so` built against a stale header.
//
// STATUS: firmed up in P2. The signatures below match what the C4
// togsim->emitc lowering (PyTorchSimFrontend/mlir/passes/lower_to_emitc.py)
// emits as `emitc.call_opaque` targets and what `mlir-translate --mlir-to-cpp`
// renders. Synchronization is event-id based: each async op is registered
// under an integer `event_id` and the matching wait passes the same id (the
// "event-id table replaces the memory-keyed tag_table" decision). Tile DRAM
// base addresses are still passed as a stub (0) until P3 wires real addresses.
// -----------------------------------------------------------------------------

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

// Emit a DMA.
//   dir       : load/store
//   arg_id    : which tensor (kernel func arg) this tile lives in
//   offset    : ELEMENT offset of this tile within that tensor, computed by the
//               producer from the loop indices (the affine address arithmetic is
//               lowered into the producer -- P3 approach A). The runtime forms
//               the DRAM address as base[arg_id] + offset*elem_bytes (only the
//               runtime knows the tensors' allocation base addresses).
//   ndim      : rank of the tile
//   dims      : ndim tile extents
//   strides   : ndim tile strides (may be null => contiguous)
//   elem_bits : element width in bits
//   is_async  : non-zero => issue-complete is the finish; the consumer must be
//               gated by an explicit togsim_memory_barrier (data arrives later).
//               Zero => blocking: the dma finishes at data-arrival.
//   tag_id    : identity of this dma's tag memref. With tag_slot it forms the
//               RUNTIME pairing key (tag_id, tag_slot) the matching
//               togsim_memory_barrier waits on -- not a compile-time id, since
//               one static dma op runs once per loop iteration.
//   tag_slot  : the SRAM tile slot this tile occupies (the producer's lowered
//               tag index, evaluated at runtime). Also the double-buffer /
//               SRAM-capacity slot. Single-buffer kernels pass 0.
//   read_bufs/n_read, write_bufs/n_write : SRAM buffer ids this op reads/writes
//   (sec 10 dataflow). The loader builds the dependency DAG by last-writer per
//   buffer.
// --- BEGIN trace-producer call formats (copied verbatim into generated trace.cpp) ---
// Each togsim_* call below lowers 1:1 to one of these free functions. Arg formats:
//   togsim_dma(ctx, dir, arg_id, offset, ndim, dims[], strides[], elem_bits,
//              is_async, tag_id, tag_slot, read_bufs[], n_read, write_bufs[], n_write)
//              dir: 0=load (MOVIN), 1=store (MOVOUT)
//   togsim_compute(ctx, tile_id, compute_type, ndim, dims[], read_bufs[], n_read,
//                  write_bufs[], n_write)   compute_type: 0=vector, 1=matmul, 2=preload
//   togsim_memory_barrier(ctx, tag_id, tag_slot, write_bufs[], n_write)
//   togsim_dispatch(ctx, tile_fn, iv[], n_iv)        // run one work-item
//   togsim_kernel(ctx, shape_args[], n_shape_args)   // producer entry point
// --- END trace-producer call formats ---
void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, int32_t tag_id, uint64_t tag_slot,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write);

// Emit a fixed-size tile compute. Cost is looked up from the precomputed
// tile_id->cycle table (annotation pass / sample-mode); `dims` are passed for
// logging and future remainder-tile handling, not to compute cost here.
//   compute_type : 0 vector / 1 matmul / 2 preload (maps to the Core unit enum;
//                  routes the op to the VPU vs the systolic array).
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);

// Explicit async-DMA sync -- the original memref.dma_wait. Pairs with its async
// togsim_dma by the RUNTIME tag slot (tag_id, tag_slot) and gates consumers on
// data-arrival (resp-complete), since an async dma's own finish is only
// issue-complete. `write_bufs` is the SRAM buffer(s) that dma loaded; the loader
// makes the barrier the last writer of them so consumers depend on it. Sync DMAs
// need no barrier (they block to data-arrival themselves).
void togsim_memory_barrier(EmitCtx* ctx, int32_t tag_id, uint64_t tag_slot,
                           const int64_t* write_bufs, int32_t n_write);

// A parallel work-item body, outlined by the producer (sec 9.3). Uniform across
// kernels: it takes the EmitCtx, the packed parallel loop indices `iv` (iv[0..
// n_iv) -- e.g. the (m,n) output-tile indices) and their count. The body emits
// the work-item's ops (init / reduction / store). One signature => one general
// dispatcher serves every kernel.
// (iv is non-const to match the `int64_t*` the EmitC producer emits; the runtime
// only reads it.)
typedef void (*togsim_tile_fn)(EmitCtx* ctx, int64_t* iv, int32_t n_iv);

// Dispatch one work-item (sec 9.3). The runtime round-robins a core from the
// pool, brackets the call with TILE_BEGIN/TILE_END (the work-item boundary), and
// invokes `fn(ctx, iv, n_iv)` -- so the work-item SCOPE is exactly the function
// call, not an implicit "ops until the next alloc" range. Core alloc + boundary
// are runtime-owned; the producer is core-count transparent (never names
// num_cores or a physical core). Independent work-items land on different cores
// -> multi-core. A general (kernel-independent) wrapper: it only forwards the
// opaque iv array to fn.
void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn,
                     int64_t* iv, int32_t n_iv);

// Entry point the loader resolves in the producer `.so`. `shape_args` carries
// the runtime values for the kernel's symbolic dimensions (in a kernel-specific
// order recorded alongside the cached `.so`); `n_shape_args` is their count.
void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args);

#ifdef __cplusplus
}  // extern "C"
#endif
