#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "SimulationConfig.h"
#include "helper/HelperFunctions.h"

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core() = default;
  bool running();
  bool can_issue(bool is_accum_tile=false);
  void issue(std::unique_ptr<Tile> tile);
  std::unique_ptr<Tile> pop_finished_tile();
  void cycle();
  bool has_memory_request();
  void pop_memory_request();
  MemoryAccess* top_memory_request() { return _request_queue.front(); }
  void push_memory_response(MemoryAccess* response);
  void print_stats();
  cycle_type get_compute_cycles() { return _stat_compute_cycle; }

 protected:
  bool can_issue_compute(std::unique_ptr<Instruction>& inst);

  /* Core id & config file */
  const uint32_t _id;   
  const SimulationConfig _config;

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _compute_end_cycle;
  cycle_type _stat_compute_cycle;
  cycle_type _stat_idle_cycle;
  cycle_type _stat_memory_cycle;
  cycle_type _compute_memory_stall_cycle;
  cycle_type _load_memory_cycle;
  cycle_type _store_memory_cycle;

  std::deque<std::unique_ptr<Tile>> _tiles;
  std::queue<std::unique_ptr<Tile>> _finished_tiles;

  std::queue<std::unique_ptr<Instruction>> _ex_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _compute_pipeline;
  std::queue<std::unique_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _st_inst_queue;

  /* Interconnect queue */
  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;
};