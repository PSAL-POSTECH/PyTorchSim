#include "Core.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _compute_end_cycle(0),
      _stat_compute_cycle(0),
      _stat_idle_cycle(0),
      _stat_memory_cycle(0),
      _compute_memory_stall_cycle(0),
      _load_memory_cycle(0),
      _store_memory_cycle(0) {
  _waiting_write_reqs = 0;
}

bool Core::can_issue(bool is_accum_tile) {
  return true; // Todo.
}

void Core::issue(std::unique_ptr<Tile> op) {
  // Todo.
  _tiles.push_back(std::move(op));
}

std::unique_ptr<Tile> Core::pop_finished_tile() {
  std::unique_ptr<Tile> result = std::make_unique<Tile>(Tile{});
  result->status = Tile::Status::EMPTY;
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::cycle() {
  _core_cycle++;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_compute_pipeline.empty();
  running = running || _waiting_write_reqs != 0;
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  running = running || !_ex_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() { return _request_queue.size() > 0; }

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(MemoryAccess *response) {
  if (response->write)
    _waiting_write_reqs--;
  delete response;
}

bool Core::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  bool result = true;
  return result;
}

void Core::print_stats() {
  spdlog::info(
      "Core [{}] : Load stall cycle {} Store stall cycle {} "
      "Total memory stall {} Idle cycle {}",
      _id, _load_memory_cycle, _store_memory_cycle,
      _stat_memory_cycle, _stat_idle_cycle);

  spdlog::info("Core [{}] : Total cycle: {}", _id, _core_cycle);
}