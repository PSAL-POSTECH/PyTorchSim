#include "Instruction.h"

Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, std::vector<size_t> tile_stride)
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(tile_size), tile_stride(tile_stride) {
  _tile_numel = 1;
  for (auto dim : tile_size)
    _tile_numel *= dim;
}

void Instruction::finish_instruction() {
  for (auto counter : child_ready_counter)
    (*counter)--;

}

void Instruction::add_child_ready_counter(size_t* counter) {
  child_ready_counter.push_back(counter);
}
