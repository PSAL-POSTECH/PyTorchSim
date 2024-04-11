#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time)
    : _config(config), _core_cycle(core_cycle), _core_time(core_time) {
  _executable_tile_queue[0] = std::deque<std::unique_ptr<Tile>>();
}

void Scheduler::schedule_model(std::unique_ptr<Model> model,
                               uint32_t sample_size) {
  _request_queue.push_back(Request{.request_id = generate_id(),
                                   .model = std::move(model),
                                   .sample_size = sample_size});
  spdlog::info("MODEL {} Scheduled, Total Request: {}",
               _request_queue.back().model->get_name(), _request_queue.size());
  refresh_status();
}


/* TODO: FIXME */ 
std::unique_ptr<Tile> Scheduler::get_tile(uint32_t core_id) {
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{});
  tile->status = Tile::Status::EMPTY;
  return tile;
}

void Scheduler::finish_tile(uint32_t core_id) {
  /* TODO: FIXME */
}

bool Scheduler::empty() { return _request_queue.empty(); }

void Scheduler::refresh_status() {
  /* TODO: FIXME */
}