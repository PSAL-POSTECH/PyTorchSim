#pragma once

#include <queue>
#include <filesystem>
#include <string>
#include "Common.h"
#include "Core.h"
#include "Dram.h"
#include "Interconnect.h"
#include "scheduler/Scheduler.h"
#include "Model.h"

namespace fs = std::filesystem;

#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3
#define IS_CORE_CYCLE(x) (x & CORE_MASK)
#define IS_DRAM_CYCLE(x) (x & CORE_MASK)
#define IS_ICNT_CYCLE(x) (x & CORE_MASK)

class Simulator {
 public:
  Simulator(SimulationConfig config);
  void schedule_graph(std::unique_ptr<TileGraph> tile_graph) { _scheduler->schedule_graph(std::move(tile_graph)); }
  void run_simulator();
 private:
  void cycle();
  void core_cycle();
  void dram_cycle();
  void icnt_cycle();
  bool running();
  void set_cycle_mask();
  uint32_t get_dest_node(MemoryAccess* access);
  SimulationConfig _config;
  uint32_t _n_cores;
  uint32_t _n_memories;

  // Components
  std::vector<std::unique_ptr<Core>> _cores;
  std::unique_ptr<Interconnect> _icnt;
  std::unique_ptr<Dram> _dram;
  std::unique_ptr<Scheduler> _scheduler;
  
  // period information (ps)
  uint64_t _core_period;
  uint64_t _icnt_period;
  uint64_t _dram_period;

  // time information (ps)
  uint64_t _core_time;
  uint64_t _icnt_time;
  uint64_t _dram_time;

  // Cycle and mask
  uint64_t _core_cycles;
  uint32_t _cycle_mask;

  // Model
  std::vector<Model> _models;
};