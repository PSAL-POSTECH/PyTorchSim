#pragma once
// togsim_trace_bridge.h -- turn a recorded trace into a TileGraph the existing
// Simulator/Core can run: one Tile per work-item (a TILE_BEGIN/TILE_END span),
// dependency edges by last-writer per SRAM buffer (sec 10).
#include <memory>

#include "TileGraph.h"
#include "togsim_loader.h"

// Build a TileGraph straight from the trace producer `so_path`.
//
// The graph is built ON DEMAND, one togsim_dispatch work-item at a time: an
// indexing pass records each dispatch's (tile fn, induction vars, core) without
// running it, and a tile's Instructions are materialized only when a core asks
// for that work-item (togsim_loader.h LazyProducer). Peak memory is therefore
// O(tiles in flight) rather than O(dispatches) -- a large 8x8 conv2d went from
// 12.4 GiB to ~0.5 GiB, with a bit-identical dependency DAG and cycle count.
//
// This is sound because a dispatch tile is dependency-closed: the bridge resets
// its writers/seeds/tag maps and finalizes its SRAM versions at every tile
// boundary, so no dependency edge and no buffer version crosses tiles.
//
// Args mirror run_producer. `name` labels the graph. Returns nullptr if the
// producer run fails, or if it emits any record outside a dispatch.
std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name);
