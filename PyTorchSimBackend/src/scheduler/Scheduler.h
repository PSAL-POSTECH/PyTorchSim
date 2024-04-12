#pragma once
#include <robin_hood.h>
#include "../Common.h"

class Scheduler {
  public:
    Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);
    virtual void schedule_model();
    virtual std::unique_ptr<Tile>& get_tile();
    virtual void finish_tile(std::unique_ptr<Tile> tile);
    virtual bool empty();

  protected:
    const cycle_type* _core_cycle;
    const uint64_t* _core_time;
    std::vector<std::unique_ptr<Tile>> _tile_queue;
    std::deque<std::vector<std::unique_ptr<Tile>>> _tile_graph;
    SimulationConfig _config;
    virtual void refresh_status();

    struct CompareTile {
    bool operator()(const std::unique_ptr<Tile>& a, const std::unique_ptr<Tile>& b) const {
      if (a->num_parent_tiles == b->num_parent_tiles) {
        return a->required_sram_size > b->required_sram_size;
      }
      return a->num_parent_tiles > b->num_parent_tiles;
    }
  };
};