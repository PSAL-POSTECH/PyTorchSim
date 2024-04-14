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

void Tile::append_instuction(std::shared_ptr<Instruction>& inst) {
  /* Move instructions */
  _instructions.push_back(inst);
}

void Tile::append_child(std::shared_ptr<Tile> child) {
  child->inc_ready_counter();
  _child_tiles.push_back(std::move(child));
}

void Tile::finish_tile() {
  for (auto& child_tile_ptr: _child_tiles)
    child_tile_ptr->dec_ready_counter();
}

void Tile::print() {
  spdlog::info("Tile: [");
  for (const auto& inst: _instructions) {
    inst->print();
  }
  spdlog::info("]");
}