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

// Build a TileGraph from a recorded trace. `path`/`name` label the graph.
std::unique_ptr<TileGraph> trace_to_tilegraph(const togsim::RunResult& run,
                                              const std::string& name);
