"""Shared vocabulary for the skeleton+API MLIR form (C1).

The trace pipeline (docs/design/togsim_cpp_trace.md) reduces a kernel's MLIR to
a *loop skeleton + API calls*: native `affine.for`/`scf.for` loops (bounds kept
as-is, symbolic preserved) plus a handful of `togsim.*` ops that stand for the
runtime API. This module is the single source of truth for those op names and
attribute keys, shared by:

  * build_skeleton (C2) -- produces the skeleton+API MLIR, and
  * togsim->emitc lowering (C4) -- rewrites each op to an `emitc.call_opaque`.

The ops are kept *unregistered* (like the existing `togsim.transfer`), so there
is no C++ dialect to register; C4 is a custom rewrite, not a registered
ConversionPass.

Grammar (each op lowers 1:1 to a `togsim_runtime.h` free function):

    "togsim.dma"(%dram_idx, %tag_idx) {         -> togsim_dma(ctx, dir, arg_id,
            dir = 0 | 1,            # LOAD|STORE      offset, ndim, dims, strides,
            dims = [..], strides = [..],                elem_bits, is_async,
            elem_bits = i32, is_async = bool,           tag_id, tag_slot,
            tag_id = i32, arg_id = i32,                 read_bufs, write_bufs)
            read_bufs = [..], write_bufs = [..]
         } : (index, index) -> ()

    "togsim.compute"() {                        -> togsim_compute(ctx, tile_id,
            tile_id = i64, compute_type = i32,          compute_type, ndim, dims,
            read_bufs = [..], write_bufs = [..]         read_bufs, write_bufs)
         } : () -> ()

    "togsim.memory_barrier"(%tag_idx) {         -> togsim_memory_barrier(ctx,
            tag_id = i32, write_bufs = [..]             tag_id, tag_slot, write_bufs)
         } : (index) -> ()

How an async dma pairs with its sync point: NOT by a compile-time id. One static
`togsim.dma` op runs once per loop iteration, each with a different RUNTIME tag
slot `%tag[%idx]`, so the pairing must be a runtime key. `togsim.dma` carries a
`tag_id` (its tag memref identity) and the runtime `%tag[%idx]` operand; the
original `memref.dma_wait` becomes an explicit `togsim.memory_barrier` carrying
the same `tag_id` + tag index. They pair at runtime by `(tag_id, tag_slot)` via
the Core's tag table (the dma signals the tag at data-arrival; the barrier waits
it). `tag_id` (which tag memref) is distinct from `tag_slot` (the SRAM tile slot,
used for the double-buffer / capacity model). A sync (non-async) dma is blocking,
so it needs no barrier. (Supersedes the earlier static `event_id` + `togsim.wait`
design, which could not express per-iteration pairing.)

Keep this in lockstep with TOGSim/include/togsim_runtime.h (TOGSIM_ABI_VERSION).
"""

# ---- op names -------------------------------------------------------------
DMA    = "togsim.dma"
COMPUTE = "togsim.compute"
MEMORY_BAR = "togsim.memory_barrier"    # explicit async-DMA sync (the original dma_wait); tag-keyed

#: every op this module owns (for matchers / DCE roots in C2).
OP_NAMES = (DMA, COMPUTE, MEMORY_BAR)

#: op name -> the togsim_runtime.h symbol C4 lowers it to.
EMITC_CALLEE = {
    DMA:     "togsim_dma",
    COMPUTE: "togsim_compute",
    MEMORY_BAR: "togsim_memory_barrier",
}

#: producer entry-point symbol the TOGSim loader resolves (see togsim_runtime.h).
ENTRY_SYMBOL = "togsim_kernel"

#: outlined per-work-item function the dispatcher hands to togsim_dispatch
#: (uniform signature (ctx, int64* iv, i32 n); see togsim_cpp_trace.md sec 9.3).
TILE_SYMBOL = "togsim_kernel_tile"

#: runtime callees emitted directly by lower_to_emitc (not skeleton ops), kept in
#: lockstep with togsim_runtime.h. DISPATCH_CALLEE is the higher-order wrapper the
#: dispatcher loop calls per work-item (round-robins a core + TILE_BEGIN/END);
#: TILE_SYMBOL is passed to it as the function pointer.
DISPATCH_CALLEE = "togsim_dispatch"

# ---- attribute keys -------------------------------------------------------
ATTR_DIR       = "dir"        # i32: DIR_LOAD | DIR_STORE
ATTR_DIMS      = "dims"       # i64 array: tile extents
ATTR_STRIDES   = "strides"    # i64 array: tile strides
ATTR_ELEM_BITS = "elem_bits"  # i32
ATTR_IS_ASYNC  = "is_async"   # bool
ATTR_TILE_ID   = "tile_id"    # i64: key into the precomputed tile_id->cycle table
ATTR_COMPUTE_TYPE = "compute_type"  # i32: 0 vector / 1 matmul / 2 preload (Core enum)
ATTR_READ_BUFS  = "read_bufs"   # i64 array: SRAM buffer ids this op reads  (sec 10 dataflow)
ATTR_WRITE_BUFS = "write_bufs"  # i64 array: SRAM buffer ids this op writes (sec 10 dataflow)
ATTR_TAG_ID    = "tag_id"     # i32: identity of the DMA's tag memref; pairs an async dma with
                              #      its memory_barrier by the RUNTIME tag slot (tag_id + tag index)
ATTR_ARG_ID    = "arg_id"     # i32: which tensor (func arg) this DMA's base is

# Must match togsim_dma_dir in togsim_runtime.h.
DIR_LOAD  = 0
DIR_STORE = 1


def is_togsim_op(op):
    """True if `op` (an Operation or a wrapping view) is one of ours."""
    name = getattr(op, "name", None)
    if name is None:
        name = getattr(getattr(op, "operation", None), "name", None)
    return name in OP_NAMES
