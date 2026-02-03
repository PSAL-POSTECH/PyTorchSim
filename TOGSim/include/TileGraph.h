#pragma once

#include <memory>
#include <map>
#include <queue>
#include <set>
#include "Tile.h"
#include "IntervalTree.h"

class TileSubGraph {
 public:
  TileSubGraph();
  void add_tile(std::shared_ptr<Tile> tile);
  void finish_tile(std::shared_ptr<Tile> tile);
  bool is_finished() { return _ready_tile_queue.empty() && _tile_set.empty(); }
  const std::shared_ptr<Tile> peek_tile();
  std::shared_ptr<Tile> get_tile();
  int get_id() { return _id; }
  void set_core_id(int core_id) { _core_id = core_id; }
  int get_core_id() { return _core_id; }
  void init_cache_plan(std::shared_ptr<IntervalTree<unsigned long long,int>> plan) { _cache_plan = plan; }
  bool is_cacheable(unsigned long long start, unsigned long long end) { return _cache_plan->findOverlapping(start, end).size() != 0; }
  struct CompareReadyTile {
    bool operator()(const std::shared_ptr<Tile>& a, const std::shared_ptr<Tile>& b) const {
      return a->get_required_sram_size() > b->get_required_sram_size();
    }
  };

 protected:
  std::priority_queue<std::shared_ptr<Tile>, std::vector<std::shared_ptr<Tile>>, CompareReadyTile> _ready_tile_queue;
  std::set<std::shared_ptr<Tile>> _tile_set;
  int _id;
  int _core_id = -1;
  static int _next_id;
  std::shared_ptr<IntervalTree<unsigned long long, int>> _cache_plan;
};

class TileGraph {
 public:
  TileGraph(std::string path, std::string name) : _path(path), _name(name), _subgraph_vec(), _cpu_graph_map() {}
  void append_subgraph(std::shared_ptr<TileSubGraph> subgraph);
  bool empty(int core_id) {
    if (_vec_index != _subgraph_vec.size()) {
        return false;
    }

    auto it = _cpu_graph_map.find(core_id);
    if (it == _cpu_graph_map.end()) {
        return true;
    }
    for (const auto& pair : it->second) {
        if (pair.second != nullptr) {
            return false;
        }
    }
    return true;
  }
  bool is_finished();
  const std::shared_ptr<Tile> peek_tile(int core_id, int slot_id);
  std::shared_ptr<Tile> get_tile(int core_id, int slot_id);
  void allocate_subgraph(int core_id, int slot_id);
  void push_range(std::string loop_idx, std::tuple<int, int, int> range) {
    _loop_index_list.push_back(loop_idx);
    _ranges.push_back(range);
  }
  std::string get_graph_path() { return _path; }
  std::string get_name() { return _name; }
  void set_arrival_time(cycle_type arrival_time) { _arrival_time = arrival_time; }
  cycle_type get_arrival_time() { return _arrival_time; }
  void set_kernel_id(unsigned int kernel_id) { _kernel_id = kernel_id; }
  unsigned int get_kernel_id() { return _kernel_id; }
  void set_start_time(cycle_type start_time) { _start_time = start_time; }
  cycle_type get_start_time() { return _start_time; }
  void init_cache_plan(IntervalTree<unsigned long long, int>::interval_vector it) {
    _cache_plan = std::make_shared<IntervalTree<unsigned long long, int>>(std::move(it));
  }
  bool StonneGraph = false;

  class Iterator {
   public:
    Iterator(const std::vector<std::tuple<int, int, int>>& ranges, const std::vector<std::string>& loop_index_list, bool end = false) :
      _ranges(ranges), _loop_index_list(loop_index_list), finished_(end) {
      if (!end)
        for (const auto& range : _ranges)
          indices_.push_back(std::get<0>(range));  // Start with the first element of each range
    }

    Iterator& operator++() {
      for (int i = indices_.size() - 1; i >= 0; --i) {
        int& current = indices_[i];
        int step = std::get<2>(_ranges[i]);
        int end = std::get<1>(_ranges[i]);

        current += step;
        if (current < end)
            return *this;

        // If the current range is exhausted, reset and move to the previous one
        current = std::get<0>(_ranges[i]);
      }

      finished_ = true;  // All ranges are exhausted
      return *this;
    }

    // Inequality operator to check if iteration is done
    bool operator!=(const Iterator& other) const {
      return finished_ != other.finished_;
    }
    std::map<std::string, int> get_indices() {
      std::map<std::string, int> result;
      for (int i=0; i<_loop_index_list.size();i++)
        result[_loop_index_list.at(i)] = indices_.at(i);
      return result;
    }
   private:
    const std::vector<std::string>& _loop_index_list;
    const std::vector<std::tuple<int, int, int>>& _ranges;
    std::vector<int> indices_;
    bool finished_ = false;
  };

  // Begin iterator
  Iterator begin() const {
    return Iterator(_ranges, _loop_index_list);
  }

  // End iterator
  Iterator end() const {
    return Iterator(_ranges, _loop_index_list, true);
  }

 private:
  int _vec_index=0;
  std::string _path;
  std::string _name = "?";
  unsigned int _kernel_id = 0;
  std::vector<std::string> _loop_index_list;
  std::vector<std::tuple<int, int, int>> _ranges;
  std::vector<std::shared_ptr<TileSubGraph>> _subgraph_vec;
  std::vector<std::shared_ptr<TileSubGraph>> _finished_subgraph_vec;
  std::map<int, std::map<int, std::shared_ptr<TileSubGraph>>> _cpu_graph_map;
  std::shared_ptr<IntervalTree<unsigned long long, int>> _cache_plan;
  cycle_type _arrival_time;
  cycle_type _start_time = 0;  // First tile issue time, 0 means not started yet
  static std::shared_ptr<Tile> null_tile;
};