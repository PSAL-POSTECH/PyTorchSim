#pragma once
// togsim_trace_bridge.h
// -----------------------------------------------------------------------------
// Bridge from the recorded trace (togsim_loader.h RunResult) to a TileGraph the
// existing Simulator/Core can run, for production cycle-equivalence (P3 task 5;
// see togsim_cpp_trace.md sec 9.9). First cut: one Tile per work-item (the span
// between two togsim_core_alloc markers), bound to that work-item's core; the
// DMA/compute records become MOVIN/MOVOUT/COMP Instructions with the RAW
// dependency edges (a compute waits the dmas its preceding waits named).
// -----------------------------------------------------------------------------
#include <memory>

#include "TileGraph.h"
#include "togsim_loader.h"

// Build a TileGraph straight from the trace producer `so_path`, streaming its
// records (nothing is retained -- the producer is replayed for the second pass;
// see togsim_loader.h run_producer_stream). Args mirror run_producer. `name`
// labels the graph. Returns nullptr if a producer run fails.
std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name);
