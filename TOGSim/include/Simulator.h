#pragma once

#include <queue>
#include <filesystem>
#include <string>
#include <yaml-cpp/yaml.h>
#include "Common.h"
#include "Core.h"
#include "SparseCore.h"
#include "Dram.h"
#include "Interconnect.h"
#include "scheduler/Scheduler.h"
#include "Model.h"

namespace fs = std::filesystem;

#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3
#define IS_CORE_CYCLE(x) (x & CORE_MASK)
#define IS_DRAM_CYCLE(x) (x & DRAM_MASK)
#define IS_ICNT_CYCLE(x) (x & ICNT_MASK)

class Simulator {
 public:
  Simulator(SimulationConfig config, YAML::Node hardware_config_yaml);
  void enqueue_graph(int partion_id, std::unique_ptr<TileGraph> tile_graph) {
    if (partion_id < 0 || static_cast<uint32_t>(partion_id) >= _config.num_partition) {
      spdlog::error("[Enqueue_graph] Invalid partition_id: {} (valid range: 0 to {}). "
                  "Total partitions: {}", partion_id, _config.num_partition - 1, _config.num_partition);
      throw std::runtime_error(
          fmt::format("[Enqueue_graph] Invalid partition_id: {} (valid range: 0 to {}). "
                    "Total partitions: {}", partion_id, _config.num_partition - 1, _config.num_partition));
    }
    _partition_scheduler.at(partion_id)->enqueue_graph(std::move(tile_graph));
  }
  void run_simulator();
  cycle_type get_core_cycle() { return _core_cycles; }
  int until(cycle_type untile_cycle);
  int get_partition_id(int core_id) { return _config.partiton_map[core_id]; }
  std::unique_ptr<Scheduler>& get_partition_scheduler(int core_id) { return _partition_scheduler.at(get_partition_id(core_id)); }
  void print_core_stat();
  void cycle();
  const SimulationConfig& get_config() const { return _config; }
  const YAML::Node& get_hardware_config_yaml() const { return _hardware_config_yaml; }
 private:
  void core_cycle();
  void dram_cycle();
  void icnt_cycle();
  bool running();
  // Spad-too-small guard: if the sim stays frozen (running() but nothing in
  // flight) past kWedgeThreshold cycles, error out and exit. Called each cycle.
  void check_frozen();
  void set_cycle_mask();
  uint32_t get_dest_node(mem_fetch *access);
  SimulationConfig _config;
  YAML::Node _hardware_config_yaml;
  uint32_t _n_cores;
  uint32_t _n_sp_cores;
  uint32_t _noc_node_per_core;
  uint32_t _n_memories;
  uint32_t _memory_req_size;
  uint32_t _slot_id;  // Double buffer slot index
  uint32_t _max_slot; // Max number of slot

  // Components
  std::vector<std::unique_ptr<Core>> _cores;
  std::unique_ptr<Interconnect> _icnt;
  std::unique_ptr<Dram> _dram;
  std::vector<std::unique_ptr<Scheduler>> _partition_scheduler;
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

  // Icnt stat
  uint64_t _nr_from_core = 0;
  uint64_t _nr_to_core = 0;
  uint64_t _nr_from_mem = 0;
  uint64_t _nr_to_mem = 0;
  cycle_type _icnt_cycle = 0;
  uint64_t _icnt_interval = 0;
};