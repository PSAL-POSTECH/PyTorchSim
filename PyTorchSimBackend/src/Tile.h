#ifndef _TILE_H
#define _TILE_H

#include <memory>
#include <deque>
#include "Instruction.h"

class Tile {
 public:
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    EMPTY,
  };

  Tile(Status status);
  Status get_status() { return _status; }
  void set_status(Status status) { _status=status; }
  size_t get_ready_counter() { return _ready_counter; }
  void inc_ready_counter(); 
  void dec_ready_counter(); 
  size_t get_required_sram_size() { return _required_sram_size; }
  void set_required_sram_size(size_t sram_size) { _required_sram_size=sram_size; }
  void append_instuctions(std::vector<std::unique_ptr<Instruction>> inst_vector);
  void append_child(std::shared_ptr<Tile> child);
  void finish_tile();
  bool is_ready() { return _ready_counter==0; }
  std::deque<std::unique_ptr<Instruction>>& get_instructions() { return _instructions; } 
  
 protected:
  Status _status = Status::EMPTY;
  size_t _required_sram_size=0;
  size_t _ready_counter=0;
  std::deque<std::unique_ptr<Instruction>> _instructions;
  std::vector<std::shared_ptr<Tile>> _child_tiles;
};

#endif