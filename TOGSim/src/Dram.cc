#include "Dram.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <spdlog/fmt/fmt.h>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

static bool is_power_of_2_u32(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

static uint32_t floor_log2_u32(uint32_t n) {
  uint32_t r = 0;
  while (n >>= 1)
    ++r;
  return r;
}

/** Smallest power of two >= n (n >= 1). */
static uint32_t next_power_of_2_u32(uint32_t n) {
  if (n <= 1)
    return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

/** Bytes/s effective GB/s and utilization % vs `peak_gbps_per_channel` (x n_ch aggregate peak). */
struct DramBwSnapshot {
  double bandwidth_gbs = 0;
  double util_avg_ch_pct = 0;
};

DramBwSnapshot make_dram_bw_snapshot(long long total_rw_transactions, uint64_t window_cycles,
                                     uint32_t n_ch, uint32_t req_size, double dram_freq_mhz,
                                     float peak_gbps_per_channel) {
  DramBwSnapshot out;
  if (window_cycles == 0 || n_ch == 0)
    return out;
  const double tx = static_cast<double>(total_rw_transactions);
  const double w = static_cast<double>(window_cycles);
  const double bytes_per_cycle = tx * static_cast<double>(req_size) / w;
  out.bandwidth_gbs = bytes_per_cycle * dram_freq_mhz / 1000.0;
  const double peak_total_gbs =
      static_cast<double>(peak_gbps_per_channel) * static_cast<double>(n_ch);
  if (peak_gbps_per_channel > 0.f && peak_total_gbs > 0.0)
    out.util_avg_ch_pct = 100.0 * out.bandwidth_gbs / peak_total_gbs;
  return out;
}

static float peak_gbps_per_channel_from_ramulator_yaml(const Ramulator::ConfigNode& cfg) {
  const Ramulator::ConfigNode controllers = cfg["memory_system"]["controllers"];
  const auto& ctrls = controllers.seq();
  if (ctrls.empty())
    throw std::runtime_error("memory_system.controllers is empty");
  const Ramulator::ConfigNode dram = ctrls[0]["dram"];
  const int ch_width = dram["channel_width"].as<int>();
  if (ch_width <= 0)
    throw std::runtime_error("invalid channel_width");
  const Ramulator::ConfigNode timing_node = dram["timing"];
  const auto& timing = timing_node.seq();
  if (timing.empty())
    throw std::runtime_error("dram.timing is empty");
  const int rate = timing[0].as<int>();
  if (rate <= 0)
    throw std::runtime_error("invalid dram.timing[0] (rate / MT/s)");

  int pseudo_ch = 1;
  const std::string impl = dram["impl"].as<std::string>("");
  if (impl == "HBM2" || impl == "HBM3") {
    const Ramulator::ConfigNode org = dram["org"];
    const Ramulator::ConfigNode org_count = org["count"];
    const auto& counts = org_count.seq();
    if (counts.size() > 1)
      pseudo_ch = std::max(1, counts[1].as<int>());
  }

  return static_cast<float>(static_cast<double>(rate) * static_cast<double>(pseudo_ch) *
                             static_cast<double>(ch_width) / 8.0 / 1000.0);
}

}  // namespace

void DramRamulator2::apply_ramulator_config_to_simulation_config(
    SimulationConfig& cfg, const std::string& ramulator_config_path,
    std::optional<uint32_t> dram_freq_mhz_stated) {
  Ramulator::ConfigNode config = Ramulator::Config::parse_config_file(ramulator_config_path);
  Ramulator::ConfigNode frontend_config;
  frontend_config.set("impl", std::string("External"));
  frontend_config.set("clock_ratio", 1u);
  config.set("frontend", frontend_config);

  float peak_gbps = 0.f;
  try {
    peak_gbps = peak_gbps_per_channel_from_ramulator_yaml(config);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("[Config/DRAM] Ramulator peak GB/s from yaml: ") + e.what() + " (" +
                             ramulator_config_path + ")");
  }

  Ramulator::IFrontEnd* fe = Ramulator::Factory::create_frontend(config);
  Ramulator::IMemorySystem* mem = Ramulator::Factory::create_memory_system(config);
  fe->connect_memory_system(mem);
  mem->connect_frontend(fe);

  const float tck_ns = mem->get_tCK();
  if (tck_ns <= 0.f) {
    fe->finalize();
    mem->finalize();
    delete fe;
    delete mem;
    throw std::runtime_error("[Config/DRAM] Ramulator probe: invalid get_tCK() for " + ramulator_config_path);
  }

  const int tx_bytes = mem->get_tx_bytes();
  if (tx_bytes <= 0) {
    fe->finalize();
    mem->finalize();
    delete fe;
    delete mem;
    throw std::runtime_error("[Config/DRAM] Ramulator probe: invalid get_tx_bytes() for " + ramulator_config_path);
  }

  fe->finalize();
  mem->finalize();
  delete fe;
  delete mem;

  cfg.dram_req_size = static_cast<uint32_t>(tx_bytes);
  cfg.dram_freq_mhz = static_cast<uint32_t>(std::lround(1000.0f / tck_ns));
  cfg.dram_bandwidth_gbps_per_channel = peak_gbps;

  if (dram_freq_mhz_stated.has_value()) {
    if (*dram_freq_mhz_stated != cfg.dram_freq_mhz) {
      throw std::runtime_error(fmt::format(
          "[Config/DRAM] ramulator2: top-level dram_freq_mhz {} does not match Ramulator timing "
          "(DRAM clock {} MHz from tCK={:.6g} ns, i.e. round(1000/tCK)); remove dram_freq_mhz to use the derived "
          "value, or align the Ramulator YAML with the top-level yml. ramulator_config_path={}",
          *dram_freq_mhz_stated, cfg.dram_freq_mhz, static_cast<double>(tck_ns), ramulator_config_path));
    }
  }
}

new_addr_type Dram::partition_dram_address(new_addr_type raw_addr) const {
  if (_req_size == 0 || _n_ch_per_partition == 0)
    return raw_addr;
  const new_addr_type tx = raw_addr >> _tx_log2;
  const new_addr_type q = tx / _n_ch_per_partition;
  return static_cast<new_addr_type>(q << _tx_log2);
}

uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_in_partition = 0;
  if (_n_ch_per_partition > 1) {
    const new_addr_type tx = static_cast<new_addr_type>(access->get_addr() >> _tx_log2);
    new_addr_type rest_high;
    unsigned init_index = 0;
    if (is_power_of_2_u32(_n_ch_per_partition)) {
      const unsigned lb = floor_log2_u32(_n_ch_per_partition);
      rest_high = tx >> lb;
      init_index = static_cast<unsigned>(tx & (_n_ch_per_partition - 1u));
    } else {
      /* gpgpu-sim "gap" channels: quotient / remainder split at txn granularity. */
      rest_high = tx / _n_ch_per_partition;
      init_index = static_cast<unsigned>(tx % _n_ch_per_partition);
    }
    /* ipoly_hash_function only implements 16/32/64 (see Hashing.cc); fold like addrdec IPOLY + mod when needed. */
    const uint32_t poly_n = next_power_of_2_u32(std::max(16u, _n_ch_per_partition));
    const uint32_t poly_use = std::min(poly_n, 64u);
    channel_in_partition =
        static_cast<uint32_t>(ipoly_hash_function(rest_high, init_index, poly_use)) % _n_ch_per_partition;
  }

  const uint32_t channel_id =
      channel_in_partition + static_cast<uint32_t>(access->get_numa_id() % _n_partitions) * _n_ch_per_partition;
  return channel_id;
}

Dram::Dram(SimulationConfig config, cycle_type* core_cycle) {
  _core_cycles = core_cycle;
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _n_partitions = config.dram_num_partitions;
  _n_ch_per_partition = config.dram_channels_per_partitions;
  _config = config;
  _tx_log2 = static_cast<int>(std::log2(_req_size));

  spdlog::info("[Config/DRAM] Total bandwidth {:.2f} GB/s, {} MHz, {} channels, {} bytes per request",
               static_cast<double>(config.max_dram_bandwidth()), config.dram_freq_mhz, _n_ch, _req_size);
  /* Initialize DRAM Channels */
  for (int ch = 0; ch < _n_ch; ch++) {
    m_to_crossbar_queue.push_back(std::queue<mem_fetch*>());
    m_from_crossbar_queue.push_back(std::queue<mem_fetch*>());
  }

  /* Initialize L2 cache */
  _m_caches.resize(_n_ch);
  if (config.l2d_type == L2CacheType::NOCACHE) {
    std::string name = "No cache";
    spdlog::info("[Config/L2Cache] No L2 cache");
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new NoL2Cache(name, _m_cache_config, ch, _core_cycles, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else if (config.l2d_type == L2CacheType::DATACACHE) {
    std::string name = "L2 cache";
    _m_cache_config.init(config.l2d_config_str);
    spdlog::info("[Config/L2Cache] Total Size: {} KB, Partition Size: {} KB, Set: {}, Assoc: {}, Line Size: {}B Sector Size: {}B",
            _m_cache_config.get_total_size_in_kb() * _n_ch, _m_cache_config.get_total_size_in_kb(),
            _m_cache_config.get_num_sets(), _m_cache_config.get_num_assoc(),
            _m_cache_config.get_line_size(), _m_cache_config.get_sector_size());
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new L2DataCache(name, _m_cache_config, ch, _core_cycles, _config.l2d_hit_latency, _config.num_cores, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else {
    spdlog::error("[Config/L2D] Invalid L2 cache type...!");
    exit(EXIT_FAILURE);
  }
}

DramRamulator2::DramRamulator2(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  /* Initialize DRAM Channels */
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<Ramulator2>(ch, _n_ch, config.dram_config_path, "Ramulator2",
                                            _config.dram_print_interval, _req_size, config.dram_freq_mhz);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

bool DramRamulator2::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (mem_fetch* req = _mem[ch]->return_queue_top())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      _mem[ch]->push(req);
      _m_caches[ch]->pop();
    }

    // From DRAM to Cache
    if (mem_fetch* req = _mem[ch]->return_queue_top()) {
      if(_m_caches[ch]->push(req))
        _mem[ch]->return_queue_pop();
    }
  }

  if (_n_ch == 0)
    return;
  const int iv = _config.dram_print_interval;
  if (iv <= 0)
    return;
  const uint64_t cc = *_core_cycles;
  if (cc % static_cast<uint64_t>(iv) != 0 || cc == 0)
    return;

  const double f_mhz = static_cast<double>(_config.dram_freq_mhz);
  const uint64_t w = static_cast<uint64_t>(iv);
  long long r_all = 0;
  long long w_all = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    const long long r = _mem[ch]->interval_reads();
    const long long wtxn = _mem[ch]->interval_writes();
    r_all += r;
    w_all += wtxn;
    const DramBwSnapshot bw = make_dram_bw_snapshot(
        r + wtxn, w, 1u, _req_size, f_mhz, _config.dram_bandwidth_gbps_per_channel);
    spdlog::trace(
        "[DRAM] channel {} | {:.2f} GB/s avg., {:.2f}% of utilization | {} reads, {} writes "
        "(interval {} cycles)",
        ch, bw.bandwidth_gbs, bw.util_avg_ch_pct, r, wtxn, w);
  }
  const DramBwSnapshot bw_all = make_dram_bw_snapshot(
      r_all + w_all, w, _n_ch, _req_size, f_mhz, _config.dram_bandwidth_gbps_per_channel);
  spdlog::info(
      "[DRAM] all {} channels combined | {:.2f} GB/s aggregate, {:.2f}% of utilization (avg. per channel) | "
      "{} reads, {} writes (interval {} cycles)",
      _n_ch, bw_all.bandwidth_gbs, bw_all.util_avg_ch_pct, r_all, w_all, w);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->reset_interval_bw_counters();
  }
}

void DramRamulator2::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

bool DramRamulator2::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

void DramRamulator2::push(uint32_t cid, mem_fetch* request) {
  const addr_type raw_addr = request->get_addr();
  const addr_type target_addr = partition_dram_address(raw_addr);
  request->set_addr(target_addr);
  m_from_crossbar_queue[cid].push(request);
}

bool DramRamulator2::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

mem_fetch* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

void DramRamulator2::print_stat() {
  spdlog::info("=== DRAM statistics ===");
  if (_n_ch == 0)
    return;

  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->finalize_once();
  }

  spdlog::trace("=== Ramulator2 stats (channels 0.. {}) ===", _n_ch - 1);
  for (int ch = 0; ch < _n_ch; ch++) {
    std::cout << "--- channel " << ch << " ---\n";
    _mem[ch]->print_stats_yaml(std::cout);
  }
  std::cout.flush();

  const uint64_t cycles = *_core_cycles;
  if (cycles == 0)
    return;
  const double f_mhz = static_cast<double>(_config.dram_freq_mhz);
  spdlog::info("[DRAM] Per-channel average bandwidth");
  long long tr_all = 0;
  long long tw_all = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    const long long tr = _mem[ch]->total_reads();
    const long long tw = _mem[ch]->total_writes();
    tr_all += tr;
    tw_all += tw;
    const DramBwSnapshot bw = make_dram_bw_snapshot(
        tr + tw, cycles, 1u, _req_size, f_mhz, _config.dram_bandwidth_gbps_per_channel);
    spdlog::info(
        "[DRAM] channel {} | {:.2f} GB/s avg., {:.2f}% of utilization | {} reads, {} writes",
        ch, bw.bandwidth_gbs, bw.util_avg_ch_pct, tr, tw);
  }
  const DramBwSnapshot bw_all = make_dram_bw_snapshot(
      tr_all + tw_all, cycles, _n_ch, _req_size, f_mhz, _config.dram_bandwidth_gbps_per_channel);
  spdlog::info(
      "[DRAM] channels 0..{} combined | {:.2f} GB/s aggregate, {:.2f}% of utilization (avg. per channel) | "
      "{} reads, {} writes",
      _n_ch - 1, bw_all.bandwidth_gbs, bw_all.util_avg_ch_pct, tr_all, tw_all);
}

void DramRamulator2::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}

void SimpleDRAM::apply_yaml_to_simulation_config(const YAML::Node& config, SimulationConfig& cfg) {
  if (!config["dram_latency"])
    throw std::runtime_error("[Config/DRAM] simple: dram_latency is required");
  cfg.dram_latency = config["dram_latency"].as<uint32_t>();

  auto yaml_get_u32 = [](const YAML::Node& n, const char* key, uint32_t def) -> uint32_t {
    if (n[key])
      return n[key].as<uint32_t>();
    return def;
  };

  cfg.dram_req_size = yaml_get_u32(config, "dram_req_size_byte", 32u);
  if (cfg.dram_req_size == 0)
    throw std::runtime_error("[Config/DRAM] simple: dram_req_size_byte must be > 0");

  const bool has_per_ch_bw = static_cast<bool>(config["dram_bandwidth_gbps_per_channel"]);
  const bool has_total_bw = static_cast<bool>(config["dram_bandwidth_gbps_total"]);
  if (has_per_ch_bw && has_total_bw)
    throw std::runtime_error(
        "[Config/DRAM] simple: set only one of dram_bandwidth_gbps_per_channel or dram_bandwidth_gbps_total");

  const bool has_bw_cap = has_per_ch_bw || has_total_bw;
  if (has_bw_cap) {
    float per_ch = 0.f;
    if (has_total_bw) {
      const float tot = config["dram_bandwidth_gbps_total"].as<float>();
      if (cfg.dram_channels == 0)
        throw std::runtime_error("[Config/DRAM] dram_channels must be > 0 for dram_bandwidth_gbps_total");
      per_ch = tot / static_cast<float>(cfg.dram_channels);
    } else {
      per_ch = config["dram_bandwidth_gbps_per_channel"].as<float>();
    }
    if (per_ch <= 0.f)
      throw std::runtime_error("[Config/DRAM] simple: dram_bandwidth_gbps_* must be > 0");
    cfg.dram_bandwidth_gbps_per_channel = per_ch;
  } else {
    cfg.dram_bandwidth_gbps_per_channel = 0.f;
  }

  if (has_bw_cap && !config["dram_freq_mhz"])
    throw std::runtime_error(
        "[Config/DRAM] simple: dram_freq_mhz is required when dram_bandwidth_gbps_per_channel or "
        "dram_bandwidth_gbps_total is set (credit refill is per simulated DRAM cycle)");
  cfg.dram_freq_mhz = yaml_get_u32(config, "dram_freq_mhz", cfg.core_freq_mhz);

  if (cfg.dram_freq_mhz == 0) {
    throw std::runtime_error("[Config/DRAM] simple: dram_freq_mhz must be > 0");
  }
}

SimpleDRAM::SimpleDRAM(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  spdlog::info("[SimpleDRAM] DRAM latency: {}", config.dram_latency);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem.push_back(std::make_unique<DelayQueue<mem_fetch*>>("SimpleDRAM", true, -1));
  }
  _latency = config.dram_latency;
  _bw_credit_bytes.assign(static_cast<size_t>(_n_ch), static_cast<double>(_req_size) * 2.0);
  if (config.dram_freq_mhz > 0 && config.dram_bandwidth_gbps_per_channel > 0.f) {
    _bytes_per_dram_cycle =
        static_cast<double>(config.dram_bandwidth_gbps_per_channel) * 1000.0 /
        static_cast<double>(config.dram_freq_mhz);
  } else {
    _bytes_per_dram_cycle = 0.;
  }
  if (config.dram_bandwidth_gbps_per_channel > 0.f)
    spdlog::info("[SimpleDRAM] peak {:.2f} GB/s total, {:.2f} GB/s per channel, {:.4f} B/cycle per channel",
                 config.max_dram_bandwidth(), config.dram_bandwidth_gbps_per_channel, _bytes_per_dram_cycle);
  else
    spdlog::info(
        "[SimpleDRAM] no bandwidth cap (latency-only); dram_latency {} cycles, dram_freq_mhz {} for tick "
        "alignment",
        config.dram_latency, config.dram_freq_mhz);
}

bool SimpleDRAM::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (!_mem[ch]->queue_empty())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

void SimpleDRAM::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    if (_bytes_per_dram_cycle > 0.0)
      _bw_credit_bytes[static_cast<size_t>(ch)] += _bytes_per_dram_cycle;

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      const double need = static_cast<double>(_req_size);
      bool admit = true;
      if (_bytes_per_dram_cycle > 0.0) {
        if (_bw_credit_bytes[static_cast<size_t>(ch)] < need)
          admit = false;
        else
          _bw_credit_bytes[static_cast<size_t>(ch)] -= need;
      }
      if (admit) {
        _mem[ch]->push(req, _latency);
        _m_caches[ch]->pop();
      }
    }

    // From DRAM to Cache
    if (_mem[ch]->arrived()) {
      mem_fetch* req = _mem[ch]->top();
      req->set_reply();
      if (_m_caches[ch]->push(req))
        _mem[ch]->pop();
    }
  }
}

void SimpleDRAM::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

bool SimpleDRAM::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

void SimpleDRAM::push(uint32_t cid, mem_fetch* request) {
  m_from_crossbar_queue[cid].push(request);
}

bool SimpleDRAM::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

mem_fetch* SimpleDRAM::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

void SimpleDRAM::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

void SimpleDRAM::print_stat() {}

void SimpleDRAM::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}
