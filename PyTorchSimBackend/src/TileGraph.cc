#include "TileGraph.h"

void TileSubGraph::add_tile(std::shared_ptr<Tile> tile) {
  if (tile->get_ready_counter() == 0) {
    _ready_tile_queue.push(std::move(tile));
  } else {
    _tile_set.insert(std::move(tile));
  }
}

void TileSubGraph::finish_tile(std::shared_ptr<Tile> tile) {
  /* TODO. */
  tile->finish_tile();
  for (auto& child_tile_ptr: tile->get_child_tile()) {
    if (child_tile_ptr->get_ready_counter())
      continue;
    /* if child is ready, add ready queue */
    _tile_set.erase(tile);
    _ready_tile_queue.push(tile);
  }
  return;
}

const std::shared_ptr<Tile> TileSubGraph::peek_tile() {
  std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
  if (_ready_tile_queue.empty())
    return ret;
  return _ready_tile_queue.top();
}

std::shared_ptr<Tile> TileSubGraph::get_tile() {
  if (_ready_tile_queue.empty()) {
    std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
    return ret;
  } else {
    std::shared_ptr<Tile> ret = _ready_tile_queue.top();
    _ready_tile_queue.pop();
    return ret;
  }
}


void TileGraph::append_subgraph(std::shared_ptr<TileSubGraph> subgraph) {
  _subgraph_vec.push_back(std::move(subgraph));
}

bool TileGraph::is_finished() {
  bool finished = _vec_index==_subgraph_vec.size();
  /* Check all outer loop is allocated */
  if (!finished)
    return finished;

  /* Check allocated subgraph is finished */
  for (const auto& pair: _cpu_graph_map) {
    if (pair.second != nullptr)
      finished &= pair.second->is_finished();
  }

  return finished;
}

const std::shared_ptr<Tile> TileGraph::peek_tile(int core_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id);
    return ret;
  } else if (_cpu_graph_map[core_id] == nullptr) {
    allocate_subgraph(core_id);
    return ret;
  }

  if (_cpu_graph_map[core_id]->is_finished()){
    allocate_subgraph(core_id);
    return ret;
  }
  return _cpu_graph_map[core_id]->peek_tile();
}

std::shared_ptr<Tile> TileGraph::get_tile(int core_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id);
    return ret;
  }

  if (_cpu_graph_map[core_id]->is_finished()) {
    allocate_subgraph(core_id);
    return ret;
  }
  return _cpu_graph_map[core_id]->get_tile();
}

void TileGraph::allocate_subgraph(int core_id) {
  if (_vec_index==_subgraph_vec.size()) {
    _cpu_graph_map[core_id] = nullptr;
    return;
  }

  std::shared_ptr<TileSubGraph> subgraph = _subgraph_vec.at(_vec_index++);
  _cpu_graph_map[core_id] = subgraph;
}