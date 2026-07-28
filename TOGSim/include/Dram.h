#ifndef DRAM_H
#define DRAM_H
#include <optional>
#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <utility>

#include "Common.h"
#include "DMA.h"
#include "ramulator2.hh"
#include "Hashing.h"
#include "Cache.h"
#include "DelayQueue.h"
#include "L2Cache.h"

class Dram {
 public:
  Dram(SimulationConfig config, cycle_type* core_cycle);
  virtual ~Dram() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual void cache_cycle() = 0;
  virtual bool is_full(uint32_t cid, mem_fetch* request) = 0;
  virtual void push(uint32_t cid, mem_fetch* request) = 0;
  virtual bool is_empty(uint32_t cid) = 0;
  virtual mem_fetch* top(uint32_t cid) = 0;
  virtual void pop(uint32_t cid) = 0;
  uint32_t get_channel_id(mem_fetch* request);
  virtual void print_stat() {}
  virtual void print_cache_stats() {};
  uint32_t get_channels_per_partition() { return _n_ch_per_partition; }
  new_addr_type partition_dram_address(new_addr_type raw_addr) const;

 protected:
  SimulationConfig _config;
  CacheConfig _m_cache_config;
  uint32_t _n_ch;
  uint32_t _n_partitions;
  uint32_t _n_ch_per_partition;
  uint32_t _req_size;
  int _tx_log2 = 0;
  cycle_type _cycles;
  cycle_type* _core_cycles;
  std::vector<DelayQueue<mem_fetch*>> m_cache_latency_queue;
  std::vector<std::queue<mem_fetch*>> m_from_crossbar_queue;
  std::vector<std::queue<mem_fetch*>> m_to_crossbar_queue;
  std::vector<std::queue<mem_fetch*>> m_to_mem_queue;
  std::vector<L2CacheBase*> _m_caches;
};

class DramRamulator2 : public Dram {
 public:
  static void apply_ramulator_config_to_simulation_config(
      SimulationConfig& cfg, const std::string& ramulator_config_path,
      std::optional<uint32_t> dram_freq_mhz_stated = std::nullopt);

  DramRamulator2(SimulationConfig config, cycle_type *core_cycle);

  virtual bool running() override;
  virtual void cycle() override;
  virtual void cache_cycle() override;
  virtual bool is_full(uint32_t cid, mem_fetch* request) override;
  virtual void push(uint32_t cid, mem_fetch* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual mem_fetch* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;
  void print_cache_stats() override;

 private:
  std::vector<std::unique_ptr<Ramulator2>> _mem;
  int _tx_ch_log2;
  int _tx_log2;
};

class SimpleDRAM: public Dram {
 public:
  static void apply_yaml_to_simulation_config(const YAML::Node& config, SimulationConfig& cfg);

  SimpleDRAM(SimulationConfig config, cycle_type *core_cycle);

  virtual bool running() override;
  virtual void cycle() override;
  virtual void cache_cycle() override;
  virtual bool is_full(uint32_t cid, mem_fetch* request) override;
  virtual void push(uint32_t cid, mem_fetch* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual mem_fetch* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;
  void print_cache_stats() override;
 private:
  int _latency = 1;
  std::vector<std::unique_ptr<DelayQueue<mem_fetch*>>> _mem;
  std::vector<double> _bw_credit_bytes;
  double _bytes_per_dram_cycle = 0.;
};

#endif