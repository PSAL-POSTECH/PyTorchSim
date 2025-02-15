#ifndef TMA_H
#define TMA_H

#include <cstdint>
#include <memory>
#include <map>
#include <vector>
#include "Instruction.h"
#include "SimulationConfig.h"
#include "Tile.h"
#include "Memfetch.h"

struct VectorCompare {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
        return a < b;
    }
};

class TMA {
 public:
  TMA(uint32_t id, uint32_t dram_req_size);

  ~TMA() {
    for (auto& subgraph: tag_used_counter) {
      for (auto& key: subgraph.second) {
        if (key.second == 0) {
          spdlog::error("Tag {} has not been used..", key.first);
        }
      }
    }
  }

  void issue_tile(std::shared_ptr<Instruction> inst);
  bool is_finished() { return _finished; }
  bool empty() { return _current_inst==nullptr; }
  void register_tag(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      tag_table[subgraph_id] = std::map<std::pair<std::string, std::vector<int>>, bool>();
      waiters[subgraph_id] = std::map<std::pair<std::string, std::vector<int>>, std::vector<std::shared_ptr<Instruction>>>();
      tag_used_counter[subgraph_id] = std::map<std::pair<std::string, std::vector<int>>, uint64_t>();
    }
    tag_table[subgraph_id][key] = false;
    waiters[subgraph_id][key] = std::vector<std::shared_ptr<Instruction>>();
  }
  void set_tag_finish(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
    if (tag_table.find(subgraph_id) == tag_table.end()) {
      throw std::runtime_error("Subgraph does not exist in tag_table");
    }
    tag_table[subgraph_id][key] = true;
  }
  bool tag_key_exist(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
    auto subgraph_it = tag_table.find(subgraph_id);
    if (subgraph_it == tag_table.end())
      return false;

    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    return key_it != key_map.end();
  }
  bool get_tag_finish(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
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
  void register_tag_waiter(int subgraph_id, const std::pair<std::string, std::vector<int>>& key, std::shared_ptr<Instruction> inst) {
    auto subgraph_it = tag_table.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    waiters[subgraph_id][key].push_back(inst);
  }
  std::vector<std::shared_ptr<Instruction>>& get_tag_waiter(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
    auto subgraph_it = tag_table.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in subgraph's tag table");
    }
    return waiters[subgraph_id][key];
  }

  void increase_tag_used_cout(int subgraph_id, const std::pair<std::string, std::vector<int>>& key) {
    auto subgraph_it = tag_used_counter.find(subgraph_id);
    auto& key_map = subgraph_it->second;
    auto key_it = key_map.find(key);
    if (key_it == key_map.end()) {
      throw std::runtime_error("Key does not exist in tag_used_counter");
    }
    tag_used_counter[subgraph_id][key]++;
  }

  std::shared_ptr<Instruction>& get_current_inst() { return _current_inst; }
  std::vector<mem_fetch*> get_memory_access();
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
  std::map<int, std::map<std::pair<std::string, std::vector<int>>, bool>> tag_table;
  std::map<int, std::map<std::pair<std::string, std::vector<int>>, std::vector<std::shared_ptr<Instruction>>>> waiters;
  std::map<int, std::map<std::pair<std::string, std::vector<int>>, uint64_t>> tag_used_counter;
};
#endif