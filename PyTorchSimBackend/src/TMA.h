#ifndef TMA_H
#define TMA_H

#include <cstdint>
#include <memory>
#include "Instruction.h"
#include "SimulationConfig.h"

typedef struct {
  uint32_t id;
  addr_type dram_address;
  uint64_t size;
  bool write;
  bool request;
  uint32_t core_id;
  Instruction* owner_instruction;
  cycle_type start_cycle;
  cycle_type dram_enter_cycle;
  cycle_type dram_finish_cycle;
} MemoryAccess;

class TMA {
 public:
  TMA(uint32_t dram_req_size);

  void issue_tile(std::shared_ptr<Instruction> inst);
  bool is_finished() { return _finished; }
  std::shared_ptr<Instruction>& get_current_inst() { return _current_inst; }
  MemoryAccess* get_memory_access();
  uint32_t generate_mem_access_id();

 protected:
  std::shared_ptr<Instruction> _current_inst;
  uint32_t _dram_req_size;
  uint32_t _tile_size_x=0;
  uint32_t _tile_size_y=0;
  size_t _tile_idx_stride=1;
  uint32_t _tile_idx;
  bool _finished=true;
};
#endif