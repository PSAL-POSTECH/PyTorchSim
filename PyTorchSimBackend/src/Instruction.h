#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

enum class Opcode { MOVIN, MOVOUT, GEMM_PRELOAD, GEMM, GEMM_WRITE, COMP, BAR };

#define SPAD_BASE 0x10000000
#define ASPAD_BASE 0x20000000
typedef uint64_t addr_type;
typedef uint64_t cycle_type;

class Instruction {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, std::vector<size_t> tile_stride)
    : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
      tile_size(tile_size), tile_stride(tile_stride) {}
  
  void finish_instruction();
  void add_child_ready_counter(size_t* counter);
  bool check_ready() { return ready_counter == 0; }
  cycle_type get_compute_cycle() { return compute_cycle; }
  // lamda function to get the dram address
  addr_type get_dram_address(int row, int col) {
    auto get_tile_address = [this](size_t i, size_t j) -> addr_type {
      return dram_addr + i * tile_size[0] * tile_stride[0] + j * tile_size[1] * tile_stride[1];
    };
    return get_tile_address(row, col);
  }
  cycle_type start_cycle;
  cycle_type finish_cycle;

 private:
  Opcode opcode;
  cycle_type compute_cycle;
  size_t ready_counter;
  std::vector<size_t*> child_ready_counter;
  std::vector<size_t> tile_size;
  std::vector<size_t> tile_stride;
  addr_type dram_addr;
};