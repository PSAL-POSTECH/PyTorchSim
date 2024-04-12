#include "Instruction.h"

void Instruction::finish_instruction() {
  for (auto counter : child_ready_counter)
    (*counter)--;
}

void Instruction::add_child_ready_counter(size_t* counter) {
  child_ready_counter.push_back(counter);
}
