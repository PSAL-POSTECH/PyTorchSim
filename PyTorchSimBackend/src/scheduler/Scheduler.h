#pragma once
#include <robin_hood.h>
#include "../Tile.h"
#include "../Common.h"
#include "../TileGraph.h"

class Scheduler {
 public:
  Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, int id);
  void schedule_graph(std::unique_ptr<TileGraph> tile_graph);
  void finish_tile(std::shared_ptr<Tile> tile) { tile->get_ownwer()->finish_tile(tile); }

  /* For other schedulers */
  virtual std::shared_ptr<Tile> get_tile(int core_id=0, int tile_id=0);
  virtual const std::shared_ptr<Tile> peek_tile(int core_id=0, int tile_id=0);
  virtual bool empty();
  virtual bool empty(int core_id);
  virtual void refresh_status();
  virtual bool check_double_buffer() { return _tile_graph.at(0)->get_subgraph_size() == 1; }

 protected:
  int _id;
  const cycle_type* _core_cycle;
  const uint64_t* _core_time;
  std::deque<std::unique_ptr<TileGraph>> _tile_graph;
  SimulationConfig _config;

  struct CompareTile {
    bool operator()(const std::shared_ptr<Tile>& a, const std::shared_ptr<Tile>& b) const {
      if (a->get_ready_counter() == b->get_ready_counter()) {
        return a->get_required_sram_size() > b->get_required_sram_size();
      }
      return a->get_ready_counter() > b->get_ready_counter();
    }
  };
};