#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time)
    : _config(config), _core_cycle(core_cycle), _core_time(core_time) {
}

void Scheduler::schedule_model() {
  spdlog::info("Tile Graph {} Scheduled", "TODO"); // TODO: tile graph id
  // _tile_graph = TileGraphScheduler->get_tile_graph();
  refresh_status();
}

std::unique_ptr<Tile>& Scheduler::get_tile() {
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{});
  tile->status = Tile::Status::EMPTY;
  if (_tile_queue.empty()) {
    return tile;
  } else {
    tile = std::move(_tile_queue.front());
    std::pop_heap(_tile_queue.begin(), _tile_queue.end(), CompareTile());
    _tile_queue.pop_back();
  }
  refresh_status();
  return tile;
}

void Scheduler::finish_tile(std::unique_ptr<Tile> tile) {
  for (auto child_tile_counter : tile->child_tiles_counter) {
    (*child_tile_counter)--;
  }
}

bool Scheduler::empty() { return _tile_graph.empty(); }

void Scheduler::refresh_status() {
  if (_tile_queue.empty()) {
    _tile_queue = std::move(_tile_graph.front());
    _tile_graph.pop_front();
    std::make_heap(_tile_queue.begin(), _tile_queue.end(), CompareTile());
  }
}