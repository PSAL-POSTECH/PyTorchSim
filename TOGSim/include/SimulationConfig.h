#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <yaml-cpp/yaml.h>

enum class CoreType { WS_MESH, STONNE };

enum class DramType { SIMPLE, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class L2CacheType { NOCACHE, DATACACHE };

struct SimulationConfig {
  /* Path to the top-level hardware YAML passed to the simulator (empty if not from a file). */
  std::string config_file_path;

  /* Core config */
  std::vector<CoreType> core_type;
  std::string stonne_config_path;
  uint32_t num_cores;
  uint32_t core_freq_mhz;
  uint32_t core_print_interval = 0;
  uint32_t num_systolic_array_per_core = 1;
  uint32_t num_stonne_per_core = 1;
  uint32_t num_stonne_port = 1;
  // Per-core VMEM/spad capacity (KB) for the trace-path DMA throttle (sec 10.x):
  // a load that would overflow the spad does not issue until a consumer frees a
  // tile. Provided by the config (the TPU configs set 16384 = 16 MB VMEM). 0 =
  // unset -> gate disabled (unlimited). Only affects trace-path instructions
  // (legacy TileGraphParser insts have alloc id -1 -> never gated).
  uint32_t core_spad_size_kb = 0;
  // SA weight-buffer depth (sec 10.x): weight tiles a systolic array holds; a
  // preload stalls until a slot frees (its matmuls finished). 2 = weight
  // double-buffer (convention default, tunable). 0 = disabled.
  uint32_t sa_weight_buffer_depth = 2;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_num_partitions = 1;
  uint32_t dram_channels_per_partitions = 0;
  uint32_t dram_freq_mhz;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  float dram_bandwidth_gbps_per_channel = 0.f;
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* L2 Cache config */
  L2CacheType l2d_type = L2CacheType::NOCACHE;
  std::string l2d_config_str;
  uint32_t l2d_hit_latency = 1;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq_mhz;
  uint32_t icnt_latency;
  uint32_t icnt_stats_print_period_cycles=0;

  /* Sheduler config */
  uint32_t num_partition=1;
  std::string scheduler_type;

  /* Core id, Partiton id mapping */
  std::map<uint32_t, uint32_t> partiton_map;

  /* Other configs */
  std::string layout;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_dram_bandwidth() const {
    if (dram_bandwidth_gbps_per_channel > 0.f)
      return dram_bandwidth_gbps_per_channel * static_cast<float>(dram_channels);
    return 0.f;
  }

  /** Resolve `path` for opening on disk: absolute paths as-is; relative paths against top-level config dir. */
  std::string resolve_against_simulation_config(const std::string& path) const {
    namespace fs = std::filesystem;
    if (path.empty())
      return path;
    fs::path p(path);
    fs::path abs = p.is_absolute() ? fs::absolute(p)
                 : !config_file_path.empty()
                     ? fs::absolute(fs::path(config_file_path).parent_path() / p)
                     : fs::absolute(p);
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(abs, ec);
    return (ec ? abs : canon).string();
  }
};