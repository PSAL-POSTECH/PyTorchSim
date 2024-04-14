#pragma once

#include <memory>
#include <map>
#include <queue>
#include <set>
#include "Tile.h"

class TileSubGraph {
 public:
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
  void append_subgraph(std::shared_ptr<TileSubGraph> subgraph);
  bool empty(int core_id) { return _vec_index==_subgraph_vec.size() && _cpu_graph_map[core_id] == nullptr; }
  bool is_finished();
  const std::shared_ptr<Tile> peek_tile(int core_id);
  std::shared_ptr<Tile> get_tile(int core_id);
  void allocate_subgraph(int core_id);

 private:
  int _vec_index=0;
  std::vector<std::shared_ptr<TileSubGraph>> _subgraph_vec;
  std::map<int, std::shared_ptr<TileSubGraph>> _cpu_graph_map;
  static std::shared_ptr<Tile> null_tile;
};