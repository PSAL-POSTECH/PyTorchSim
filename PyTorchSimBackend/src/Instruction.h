#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <set>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

enum class Opcode { MOVIN, MOVOUT, GEMM_PRELOAD, GEMM, GEMM_WRITE, COMP, BAR };

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

std::string opcode_to_string(Opcode opcode);

class Instruction {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, std::vector<size_t> tile_stride);
  void finish_instruction();
  void add_child(std::shared_ptr<Instruction> child);
  bool check_ready() { return ready_counter == 0; }
  const Opcode get_opcode() { return opcode; }
  bool is_dma_read() { return opcode == Opcode::MOVIN; }
  bool is_dma_write() { return opcode == Opcode::MOVOUT; }
  bool is_ready() { return ready_counter == 0; }
  void inc_ready_counter() { ready_counter++; }
  void dec_ready_counter() { ready_counter--; }
  size_t get_tile_numel() { return _tile_numel; }
  size_t get_precision() { return _precision; }
  void inc_waiting_request();
  void dec_waiting_request();
  size_t get_waiting_request() { return _nr_waiting_request; }
  std::vector<size_t>& get_tile_size() { return tile_size; }
  cycle_type get_compute_cycle() { return compute_cycle; }
  void print();
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
  std::set<std::shared_ptr<Instruction>> child_inst;
  std::vector<size_t> tile_size;
  std::vector<size_t> tile_stride;
  size_t _tile_numel;
  size_t _nr_waiting_request=0;
  size_t _precision;
  addr_type dram_addr;
};