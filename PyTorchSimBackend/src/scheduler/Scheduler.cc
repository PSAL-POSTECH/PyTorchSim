#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, int id)
    : _id(id), _config(config), _core_cycle(core_cycle), _core_time(core_time) {
}

void Scheduler::schedule_graph(std::unique_ptr<TileGraph> tile_graph) {
  spdlog::info("[Scheduler {}] Tile Graph {} Scheduled", _id, "FIFO"); // TODO: tile graph id
  // _tile_graph = TileGraphScheduler->get_tile_graph();
  _tile_graph.push_back(std::move(tile_graph));
  refresh_status();
}

const std::shared_ptr<Tile> Scheduler::peek_tile(int core_id, int tile_id) {
  if (_tile_graph.empty() || _tile_graph.at(0)->get_arrival_time() > *_core_cycle)
    return std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (check_double_buffer())
    tile_id = 0;
  return _tile_graph.at(0)->peek_tile(core_id, tile_id);
}

std::shared_ptr<Tile> Scheduler::get_tile(int core_id, int tile_id) {
  std::shared_ptr<Tile> tile = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (empty(core_id)) {
    return tile;
  } else {
    if (check_double_buffer())
      tile_id = 0;
    tile = std::move(_tile_graph.at(0)->get_tile(core_id, tile_id));
  }
  refresh_status();
  return tile;
}

bool Scheduler::empty() {
  if (_tile_graph.empty())
    return true;
  return false;
}

bool Scheduler::empty(int core_id) {
  if (_tile_graph.empty())
    return true;
  return _tile_graph.at(0)->empty(core_id);
}

void Scheduler::refresh_status() {
  if (_tile_graph.empty())
    return;

  /* Remove finished request */
  if (_tile_graph.at(0)->is_finished()) {
    spdlog::info("Kernel {} finish at {}",
                 _tile_graph.at(0)->get_name(), *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _tile_graph.at(0)->get_arrival_time());
    _tile_graph.pop_front();
  }
}