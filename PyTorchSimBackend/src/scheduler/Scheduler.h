#pragma once
#include <robin_hood.h>
#include "../Common.h"
#include "../Model.h"

typedef struct {
  uint32_t request_id;
  std::unique_ptr<Model> model;
  uint32_t sample_size;
} Request;

class Scheduler {
  public:
    Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);
    virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size);
    virtual std::unique_ptr<Tile> get_tile(uint32_t core_id);
    virtual void finish_tile(uint32_t core_id);
    virtual bool empty();

  protected:
    const cycle_type* _core_cycle;
    const uint64_t* _core_time;
    std::deque<Request> _request_queue;
    std::map<uint32_t, std::deque<std::unique_ptr<Tile>>> _executable_tile_queue;
    SimulationConfig _config;
    virtual void refresh_status();
};