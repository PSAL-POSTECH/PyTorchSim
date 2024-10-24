#pragma once

#include <memory>
#include <map>
#include <queue>
#include <set>
#include "Tile.h"

class TileSubGraph {
 public:
  TileSubGraph();
  void add_tile(std::shared_ptr<Tile> tile);
  void finish_tile(std::shared_ptr<Tile> tile);
  bool is_finished() { return _ready_tile_queue.empty() && _tile_set.empty(); }
  const std::shared_ptr<Tile> peek_tile();
  std::shared_ptr<Tile> get_tile();
  struct CompareReadyTile {
    bool operator()(const std::shared_ptr<Tile>& a, const std::shared_ptr<Tile>& b) const {
      return a->get_required_sram_size() > b->get_required_sram_size();
    }
  };

 protected:
  std::priority_queue<std::shared_ptr<Tile>, std::vector<std::shared_ptr<Tile>>, CompareReadyTile> _ready_tile_queue;
  std::set<std::shared_ptr<Tile>> _tile_set;
};

class TileGraph {
 public:
  TileGraph(std::string name) : _name(name), _subgraph_vec(), _cpu_graph_map() {}
  void append_subgraph(std::shared_ptr<TileSubGraph> subgraph);
  bool empty(int core_id) { return _vec_index==_subgraph_vec.size() && _cpu_graph_map[core_id][0] == nullptr && _cpu_graph_map[core_id][1] == nullptr; }
  bool is_finished();
  const std::shared_ptr<Tile> peek_tile(int core_id, int tile_id);
  std::shared_ptr<Tile> get_tile(int core_id, int tile_id);
  uint32_t get_subgraph_size() { return _subgraph_vec.size(); }
  void allocate_subgraph(int core_id, int tile_id);
  void push_range(std::string loop_idx, std::tuple<int, int, int> range) {
    _loop_index_list.push_back(loop_idx);
    _ranges.push_back(range);
  }
  std::string get_name() { return _name; }
  void set_arrival_time(cycle_type arrival_time) { _arrival_time = arrival_time; }
  cycle_type get_arrival_time() { return _arrival_time; }

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
  std::string _name;
  std::vector<std::string> _loop_index_list;
  std::vector<std::tuple<int, int, int>> _ranges;
  std::vector<std::shared_ptr<TileSubGraph>> _subgraph_vec;
  std::map<int, std::map<int, std::shared_ptr<TileSubGraph>>> _cpu_graph_map;
  cycle_type _arrival_time;
  static std::shared_ptr<Tile> null_tile;
};