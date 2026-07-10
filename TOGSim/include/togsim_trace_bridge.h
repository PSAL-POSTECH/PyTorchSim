#pragma once
// togsim_trace_bridge.h -- turn a recorded trace into a TileGraph the existing
// Simulator/Core can run: one Tile per work-item (a TILE_BEGIN/TILE_END span),
// dependency edges by last-writer per SRAM buffer (sec 10).
#include <memory>

#include "TileGraph.h"
#include "togsim_loader.h"

// Build a TileGraph from a recorded trace. `name` labels the graph.
std::unique_ptr<TileGraph> trace_to_tilegraph(const togsim::RunResult& run,
                                              const std::string& name);
