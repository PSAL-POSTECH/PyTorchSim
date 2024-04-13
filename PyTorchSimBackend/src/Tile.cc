#include "Tile.h"

Tile::Tile(Status status) {
  _status = status;
}

void Tile::inc_ready_counter() {
  _ready_counter++;
}

void Tile::dec_ready_counter() {
  if (_ready_counter==0) {
    spdlog::error("Tile ready counter is already 0...");
    exit(EXIT_FAILURE);
  }
  _ready_counter--;
}

void Tile::append_instuctions(std::vector<std::unique_ptr<Instruction>>& inst_vector) {
  /* Move instructions */
  _instructions.insert(
    _instructions.end(),
    std::make_move_iterator(inst_vector.begin()),
    std::make_move_iterator(inst_vector.end())
  );
}

void Tile::append_child(std::shared_ptr<Tile> child) {
  _child_tiles.push_back(std::move(child));
}

void Tile::finish_tile() {
  for (auto& child_tile_ptr: _child_tiles)
    child_tile_ptr->dec_ready_counter();
}