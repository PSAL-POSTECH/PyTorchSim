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

// Bump whenever the signatures below change incompatibly. TOGSim refuses to load
// a producer whose embedded version (a `togsim_producer_abi_version` symbol, or
// a value passed at the entry point) does not match.
//   v1 -> v2 (P2): dma takes an event_id and returns void (was: returns a
//                  handle); togsim_kernel shape_args is non-const to match the
//                  emitc/mlir-to-cpp output.
//   v2 -> v3 (P3): add togsim_dispatch (work-item boundary + core binding) and
//                  togsim_wait_all (join / barrier).
//   v3 -> v4 (P3): togsim_dma takes (arg_id, element offset) instead of a
//                  precomputed base_addr; the producer lowers the address
//                  arithmetic and the runtime adds the tensor base.
//   v4 -> v5 (P3): event handles. togsim_dma RETURNS a fresh handle (drops the
//                  event_id arg); the producer parks it in a heap event buffer
//                  (togsim_event_alloc/free) and togsim_wait takes the handle.
//   v5 -> v6 (P3): replace togsim_dispatch with togsim_core_alloc (returns a
//                  core id; no free) -- the runtime owns the core pool, num_cores
//                  is never baked into the producer.
//   v6 -> v7 (P3): togsim_dma takes a tag_slot (SRAM tile slot) for the runtime's
//                  double-buffer / SRAM-capacity model.
//   v7 -> v8 (P3): togsim_compute takes a compute_type (vector/matmul/preload) so
//                  the Core routes it to the right compute unit.
//   v8 -> v9 (P3 sec10): togsim_dma/compute take read_bufs/write_bufs (SRAM buffer
//                  ids); the loader builds an explicit dependency DAG by
//                  last-writer per buffer (replaces in-order/tag dependencies).
//   v9 -> v10 (P3 sec10.7): add togsim_compute_barrier (the explicit compute fence
//                  before a store; loader -> COMPUTE_BAR instruction).
//   v10 -> v11 (P3 sec10): replace the static event-id pairing with the RUNTIME
//                  tag slot. togsim_dma takes a tag_id (its tag memref identity)
//                  and returns void; the original dma_wait becomes an explicit
//                  togsim_memory_barrier(tag_id, tag_slot, write_bufs) that pairs
//                  with its async dma by the runtime (tag_id, tag_slot) -- one
//                  static dma op runs once per loop iteration with a different
//                  %tag[%idx], so only a runtime key can pair them. Drops
//                  togsim_wait/signal/wait_all/event_alloc/event_free + the
//                  togsim_event handle (no compile-time pairing token).
//   v11 -> v12 (P3 sec9.3): replace the bare togsim_core_alloc marker with a
//                  higher-order togsim_dispatch(ctx, tile_fn, iv, n_iv) wrapper.
//                  The producer outlines each parallel work-item into a uniform
//                  togsim_kernel_tile(ctx, iv, n) and the dispatcher loop hands it
//                  to togsim_dispatch, which round-robins a core and brackets the
//                  call with TILE_BEGIN/TILE_END. The work-item scope is now the
//                  function call itself (no implicit "until the next core_alloc"
//                  range); one general dispatcher serves every kernel (uniform
//                  iv-array ABI). Core alloc + the begin/end boundary are
//                  runtime-owned.
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
