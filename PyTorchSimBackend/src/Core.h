#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "Tile.h"
#include "SimulationConfig.h"
#include "TMA.h"

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core() = default;
  bool running();
  bool can_issue(const std::shared_ptr<Tile>& op);
  void issue(std::shared_ptr<Tile> tile);
  std::shared_ptr<Tile> pop_finished_tile();
  void cycle();
  void compute_cycle();
  void dma_cycle();
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
  size_t _sram_size;
  size_t _remain_sram_size;

  /* TMA Unit */
  TMA _tma;

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _stat_compute_cycle;
  cycle_type _stat_idle_cycle;
  cycle_type _stat_tma_cycle;
  cycle_type _stat_issued_cycle;
  cycle_type _compute_memory_stall_cycle;

  std::deque<std::shared_ptr<Tile>> _tiles;
  std::queue<std::shared_ptr<Tile>> _finished_tiles;

  std::queue<std::unique_ptr<Instruction>> _compute_pipeline;
  std::queue<std::unique_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _st_inst_queue;

  std::vector<std::unique_ptr<Instruction>> _dma_waiting_queue;
  /* Interconnect queue */
  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;
};