#pragma once
// togsim_trace_bridge.h -- turn a recorded trace into a TileGraph the existing
// Simulator/Core can run: one Tile per work-item (a TILE_BEGIN/TILE_END span),
// dependency edges by last-writer per SRAM buffer (sec 10).
#include <memory>

#include "TileGraph.h"
#include "togsim_loader.h"

// Build a TileGraph straight from the trace producer `so_path`. The graph is built
// ON DEMAND, one togsim_dispatch work-item at a time (see togsim_loader.h): peak
// memory is O(tiles in flight), not O(dispatches). Sound because a tile is
// dependency-closed; SRAM buffer VERSIONS do cross tiles though.
//
// `name` labels the graph. Returns nullptr if the producer fails to load or run.
std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name);
