#pragma once
#include <fstream>
#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <list>
#include <numeric>

#include <set>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

enum class Opcode { MOVIN, MOVOUT, COMP, BAR, COUNT};

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

std::string opcode_to_string(Opcode opcode);

class Instruction : public std::enable_shared_from_this<Instruction> {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, size_t precision, std::vector<int> &idx_list,
              std::vector<int> &stride_list,  std::vector<int> tag_idx_list, std::vector<int> tag_stride_list,
              std::vector<int> accum_tag_idx_list, std::vector<int> loop_size_list);
  void finish_instruction();
  void add_child(std::shared_ptr<Instruction> child);
  bool check_ready() { return ready_counter == 0; }
  const Opcode get_opcode() { return opcode; }
  bool is_dma_read() { return opcode == Opcode::MOVIN; }
  bool is_dma_write() { return opcode == Opcode::MOVOUT; }
  bool is_async_dma() { return _is_async_dma; }
  bool is_indirect_mode() { return _is_indirect_mode; }
  std::string get_indirect_index_path() { return _indirect_index_path; }
  bool is_ready() { return ready_counter == 0; }
  void inc_ready_counter() { ready_counter++; }
  void dec_ready_counter() {
    assert(ready_counter!=0);
    ready_counter--;
    if (!ready_counter && _owner_ready_queue_ref != nullptr) {
      _owner_ready_queue_ref->push_back(shared_from_this());
    }
  }
  size_t get_tile_numel() { return _tile_numel; }
  size_t get_precision() { return _precision; }
  void inc_waiting_request();
  void dec_waiting_request();
  size_t get_waiting_request() { return _nr_waiting_request; }
  std::vector<size_t>& get_tile_size() { return tile_size; }
  void set_overlapping_cycle(cycle_type cycle) { overlapping_cycle = cycle; }
  cycle_type get_overlapping_cycle() { return overlapping_cycle; }
  cycle_type get_compute_cycle() { return compute_cycle; }
  void set_compute_cycle(cycle_type cycle) { compute_cycle = cycle; }
  void set_indirect_index_path(std::string indirect_path) { _is_indirect_mode=true; _indirect_index_path=indirect_path; }
  void print();
  std::shared_ptr<std::set<addr_type>> get_dram_address(addr_type dram_req_size);
  std::vector<addr_type> get_trace_address() { return _trace_address; }
  bool load_indirect_index(const std::string& path, uint64_t*& indirect_index, const std::vector<uint64_t>& tile_size);
  void set_trace_address(std::vector<addr_type>& trace_address) { _trace_address = trace_address; }
  size_t get_free_sram_size() { return _free_sram_size; }
  void adjust_dram_address() {
    int offset = std::inner_product(_idx_list.begin(), _idx_list.end(), _stride_list.begin(), 0);
    dram_addr += offset * _precision;
  }
  addr_type get_base_dram_address() { return dram_addr; }
  void set_free_sram_size(size_t sram_size) { _free_sram_size=sram_size; }
  void* get_owner() { return _owner; }
  void set_owner(void *owner) { _owner = owner;}
  void set_owner_ready_queue(std::list<std::shared_ptr<Instruction>>* q) { _owner_ready_queue_ref = q; }
  void set_compute_type(int type) { _compute_type = type; }
  int get_compute_type() { return _compute_type; }
  void set_numa_id(int numa_id) { _numa_id = numa_id; }
  uint32_t get_numa_id() { return _numa_id; }
  std::vector<int>& get_idx_list() { return _idx_list; }
  std::vector<int>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<int>& get_tag_stride_list() { return _tag_stride_list; }
  std::vector<int>& get_tag_id() { return _tag_key; }
  void set_addr_name(std::string name, int id) { _addr_name = name; _addr_id = id; }
  std::string get_addr_name() { return _addr_name; }
  int get_addr_id() { return _addr_id; }
  void set_nr_inner_loop(int nr) { _nr_inner_loop = nr; }
  int get_nr_inner_loop() { return _nr_inner_loop; }
  void set_is_async(bool is_async) { _is_async_dma = is_async; }
  void prepare_tag_key();
  bool is_sparse_inst() { return _is_sparse_inst; }
  void set_sparse_state(bool state) { _is_sparse_inst = state; }
  std::set<std::shared_ptr<Instruction>>& get_child_inst() { return child_inst; }

  cycle_type start_cycle;
  cycle_type finish_cycle;
  cycle_type bubble_cycle=0;

  bool finished=false;
  int subgraph_id;
 private:
  void *_owner = nullptr;
  std::list<std::shared_ptr<Instruction>>* _owner_ready_queue_ref = nullptr;
  Opcode opcode;
  cycle_type compute_cycle;
  cycle_type overlapping_cycle;
  size_t ready_counter;
  std::set<std::shared_ptr<Instruction>> child_inst;
  std::vector<size_t> tile_size;
  size_t _tile_numel;
  size_t _nr_waiting_request=0;
  size_t _precision=0;
  size_t _free_sram_size=0;
  addr_type dram_addr;
  uint32_t _numa_id = 0; // For DMA instruction
  int _compute_type = 0;
  std::vector<int> _idx_list;
  std::vector<int> _stride_list;
  std::vector<int> _tag_idx_list;
  std::vector<int> _tag_stride_list;
  std::vector<int> _tag_key;
  std::vector<int> _accum_tag_idx_list;
  std::vector<int> _loop_size_list;
  std::vector<addr_type> _trace_address;
  std::string _addr_name;
  int _addr_id;
  int _nr_inner_loop = 0;
  bool _is_async_dma=false;
  bool _is_indirect_mode=false;
  bool _is_sparse_inst=false;
  std::string _indirect_index_path="";
};