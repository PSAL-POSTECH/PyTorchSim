#ifndef DMA_H
#define DMA_H

#include <cstdint>
#include <memory>
#include <queue>
#include <map>
#include <vector>
#include "Instruction.h"
#include "SimulationConfig.h"
#include "Tile.h"
#include "Memfetch.h"

struct VectorCompare {
    bool operator()(const std::vector<int64_t>& a, const std::vector<int64_t>& b) const {
        return a < b;
    }
};

class DMA {
 public:
  DMA(uint32_t id, uint32_t dram_req_size, bool l2_datacache_enabled);

  void issue_tile(std::shared_ptr<Instruction> inst);
  bool is_finished() { return _finished; }
  bool empty() { return _current_inst==nullptr; }
  void register_tag(int subgraph_id, std::vector<int64_t>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      tag_table[subgraph_id] = std::map<std::vector<int64_t>, uint32_t>();
      waiters[subgraph_id] = std::map<std::vector<int64_t>, std::vector<std::shared_ptr<Instruction>>>();
    }
    tag_table[subgraph_id][key] = 0;
    waiters[subgraph_id][key] = std::vector<std::shared_ptr<Instruction>>();
  }
  void set_tag_finish(int subgraph_id, std::vector<int64_t>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table[subgraph_id][key] = 1;
  }

  void set_tag_sparse(int subgraph_id, std::vector<int64_t>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table[subgraph_id][key] = -1;
  }

  void mark_tag_used(int subgraph_id, std::vector<int64_t>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    } else if (!tag_table[subgraph_id][key]) {
      throw std::runtime_error("Tag is not ready but freed");
    }
    tag_table[subgraph_id][key] += 1;
  }

  void check_table() {
    for (const auto& entry: tag_table) {
      auto subgraph_id = entry.first;
      for (const auto& tag_entry: tag_table[subgraph_id]) {
        const std::vector<int64_t>& tag_key = tag_entry.first;
        uint32_t value = tag_entry.second;
        if (value == 1) {
          spdlog::debug("[Tag Table][{}] Unused tag found: (key={}, val={})",
            subgraph_id, fmt::format("[{}]", fmt::join(tag_key, ", ")), value);
        }
      }
    }
  }

  bool tag_key_exist(int subgraph_id, std::vector<int64_t>& key) {
    auto subgraph_it = tag_table.find(subgraph_id);
    if (subgraph_it == tag_table.end())
      return false;

    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    return key_it != key_map.end();
  }
  uint32_t get_tag_finish(int subgraph_id, std::vector<int64_t>& key) {
    auto subgraph_it = tag_table.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    return tag_table[subgraph_id][key];
  }
  void erase_tag_table(int subgraph_id) {
    auto subgraph_it = tag_table.find(subgraph_id);
    if (subgraph_it == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table.erase(subgraph_id);
    waiters.erase(subgraph_id);
  }
  void register_tag_waiter(int subgraph_id, std::vector<int64_t>& key, std::shared_ptr<Instruction> inst) {
    auto subgraph_it = tag_table.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    waiters[subgraph_id][key].push_back(inst);
  }
  std::vector<std::shared_ptr<Instruction>>& get_tag_waiter(int subgraph_id, std::vector<int64_t>& key) {
    auto subgraph_it = tag_table.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    return waiters[subgraph_id][key];
  }

  std::shared_ptr<Instruction>& get_current_inst() { return _current_inst; }
  std::shared_ptr<std::vector<mem_fetch*>> get_memory_access(cycle_type core_cycle, int nr_req);
  uint32_t generate_mem_access_id();
  const uint32_t get_max_dim() { return _max_dim; }

 protected:
  uint32_t _id;
  const uint32_t _max_dim = 4;
  std::shared_ptr<Instruction> _current_inst;
  uint32_t _dram_req_size;
  uint32_t _tile_size_x=0;
  uint32_t _tile_size_y=0;
  size_t _tile_idx_stride=1;
  uint32_t _tile_idx;
  bool _finished=true;
  bool _l2_datacache_enabled = false;
  std::map<int, std::map<std::vector<int64_t>, uint32_t>> tag_table;
  std::map<int, std::map<std::vector<int64_t>, std::vector<std::shared_ptr<Instruction>>>> waiters;
  std::queue<mem_fetch*> _pending_accesses;
  bool _generated_once = false;
};
#endif
