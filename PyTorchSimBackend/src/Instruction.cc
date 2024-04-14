#include "Instruction.h"

std::string opcode_to_string(Opcode opcode) {
    switch (opcode) {
        case Opcode::MOVIN:        return "MOVIN";
        case Opcode::MOVOUT:       return "MOVOUT";
        case Opcode::GEMM_PRELOAD: return "GEMM_PRELOAD";
        case Opcode::GEMM:         return "GEMM";
        case Opcode::GEMM_WRITE:   return "GEMM_WRITE";
        case Opcode::COMP:         return "COMP";
        case Opcode::BAR:          return "BAR";
        default:                   return "Unknown";
    }
}

Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, std::vector<size_t> tile_stride)
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(tile_size), tile_stride(tile_stride) {
  _tile_numel = 1;
  for (auto dim : tile_size)
    _tile_numel *= dim;
}

void Instruction::finish_instruction() {
  for (auto& counter : child_inst)
    counter->dec_ready_counter();
}

void Instruction::add_child(std::shared_ptr<Instruction> child) {
  child->inc_ready_counter();
  child_inst.insert(child);
}

void Instruction::inc_waiting_request() {
  _nr_waiting_request++;
}

void Instruction::dec_waiting_request() {
  assert(_nr_waiting_request!=0);
  _nr_waiting_request--;
}

void Instruction::print() {
  spdlog::info("{}", opcode_to_string(opcode));
}