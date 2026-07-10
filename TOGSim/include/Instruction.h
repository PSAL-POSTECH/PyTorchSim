#pragma once
#include <fstream>
#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <list>
#include <numeric>

#include <array>
#include <set>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

// MEMORY_BAR: the DMA/memory barrier (waits a DMA tag in the tag table).
enum class Opcode { MOVIN, MOVOUT, COMP, MEMORY_BAR, COUNT};

// A dependency edge releases its consumer on one of the producer's lifecycle
// events: ISSUE (occupancy -- the consumer overlaps the producer on the SA
// pipeline) or DONE (latency -- the consumer needs the producer's result).
enum class DepEvent : uint8_t { ISSUE = 0, DONE = 1, COUNT = 2 };

// One weight slot on systolic array `sa` (sec 10.4). A preload sets refcount =
// the matmuls reusing the weight; each frees it at its streaming-end, the last
// one releases the slot. Shared (shared_ptr) by the preload's matmul consumers.
struct WeightToken { int sa; int refcount; };

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

std::string opcode_to_string(Opcode opcode);
std::string format_tag_key_list_hex(const std::vector<int64_t>& tag_keys);

class Instruction;
// Order dependents by creation order (_global_inst_id), NOT by heap address.
// std::set<shared_ptr<...>>'s default comparator orders by pointer value, which
// only looked stable because the whole TileGraph was allocated up front in one
// monotonically growing run: fire() then released dependents in creation order
// by accident. Under any other heap layout the release order -- and with it the
// issue order and the reported cycle count -- changes. The instruction id is
// intrinsic, so ordering by it makes the release order canonical.
struct DepLess {
  bool operator()(const std::shared_ptr<Instruction>& a,
                  const std::shared_ptr<Instruction>& b) const;
};

class Instruction : public std::enable_shared_from_this<Instruction> {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, std::vector<int> tile_stride, size_t elem_bits,
              std::vector<int64_t> tag_idx_list, std::vector<int64_t> tag_stride_list,
              std::vector<int64_t> accum_tag_idx_list);
  Instruction(Opcode opcode);
  void finish_instruction();
  // Subscribe `c` to this op's `on` event (ISSUE=occupancy, DONE=latency). The set
  // dedups, so ready_counter is bumped only on a new edge (a producer writing
  // several buffers one consumer reads links the pair once per buffer).
  void add_dep(std::shared_ptr<Instruction> c, DepEvent on) {
    if (_deps[static_cast<size_t>(on)].insert(c).second) c->inc_ready_counter();
  }
  // Release every subscriber of `e` (decrement its ready_counter) and clear.
  void fire(DepEvent e) {
    for (auto& c : _deps[static_cast<size_t>(e)]) c->dec_ready_counter();
    _deps[static_cast<size_t>(e)].clear();
  }
  const std::set<std::shared_ptr<Instruction>, DepLess>& get_deps(DepEvent e) {
    return _deps[static_cast<size_t>(e)];
  }
  void set_assigned_sa(int s) { _assigned_sa = s; }
  int get_assigned_sa() const { return _assigned_sa; }
  void set_weight_token(const std::shared_ptr<WeightToken>& t) { _weight_token = t; }
  const std::shared_ptr<WeightToken>& get_weight_token() const { return _weight_token; }
  // Trace-only: which work-item (togsim_dispatch tile) this op belongs to, for
  // grouping/coloring in the timeline. Set by the bridge per TILE_BEGIN.
  void set_tile_group(int g) { _tile_group = g; }
  int get_tile_group() const { return _tile_group; }
  bool check_ready() { return ready_counter == 0; }
  const Opcode get_opcode() { return opcode; }
  bool is_dma_read() { return opcode == Opcode::MOVIN; }
  bool is_dma_write() { return opcode == Opcode::MOVOUT; }
  bool is_dma_instruction() const { return opcode == Opcode::MOVIN || opcode == Opcode::MOVOUT; }
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
      if (_owner_dirty_ref) *_owner_dirty_ref = true;   // ready set grew -> re-arm the owning Core's issue scan
    }
  }
  size_t get_tile_numel() { return _tile_numel; }
  size_t get_elem_bits() const { return _elem_bits; }
  void inc_waiting_request();
  void dec_waiting_request();
  size_t get_waiting_request() { return _nr_waiting_request; }
  // trace: log only the FIRST DRAM response of a load (when data starts arriving).
  bool got_first_response() const { return _got_first_response; }
  void mark_first_response() { _got_first_response = true; }
  std::vector<size_t>& get_tile_size() { return tile_size; }
  std::vector<int>& get_tile_stride() { return tile_stride; }
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
  addr_type get_base_dram_address() { return dram_addr; }
  void* get_owner() { return _owner; }
  void set_owner(void *owner) { _owner = owner;}
  void set_owner_ready_queue(std::list<std::shared_ptr<Instruction>>* q) { _owner_ready_queue_ref = q; }
  // Points at the owning Core's _issue_dirty; set when the tile is issued to a core
  // (Core::issue) so a dep-resolved enqueue re-arms only that core's issue scan.
  void set_owner_dirty(bool* d) { _owner_dirty_ref = d; }
  void set_compute_type(int type) { _compute_type = type; }
  int get_compute_type() { return _compute_type; }
  void set_numa_id(int numa_id) { _numa_id = numa_id; }
  uint32_t get_numa_id() { return _numa_id; }
  std::vector<int64_t>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<int64_t>& get_tag_stride_list() { return _tag_stride_list; }
  std::vector<int64_t>& get_tag_id() { return _tag_key; }
  void set_addr_name(std::string name, int64_t id) { _addr_name = name; _addr_id = id; }
  std::string get_addr_name() { return _addr_name; }
  int64_t get_addr_id() { return _addr_id; }
  void set_nr_inner_loop(int nr) { _nr_inner_loop = nr; }
  int get_nr_inner_loop() { return _nr_inner_loop; }
  void set_is_async(bool is_async) { _is_async_dma = is_async; }
  void prepare_tag_key();
  bool is_sparse_inst() { return _is_sparse_inst; }
  void set_sparse_state(bool state) { _is_sparse_inst = state; }
  uint64_t get_global_inst_id() const { return _global_inst_id; }

  // SRAM-capacity model (sec 10.4), filled by the bridge and enforced by Core: a
  // load fills buffer-version `_sram_alloc_id` (-1 = untracked), and a version's
  // last reader frees it on issue via `_sram_release_allocs`.
  void set_sram_alloc(int64_t id) { _sram_alloc_id = id; }
  int64_t get_sram_alloc() const { return _sram_alloc_id; }
  void add_sram_release(int64_t id) { _sram_release_allocs.push_back(id); }
  const std::vector<int64_t>& get_sram_release() const { return _sram_release_allocs; }
  // bytes this instruction's buffer occupies in the spad. A DMA derives it from
  // the tile it moves; a compute output gets it set explicitly by the bridge (the
  // buffer's size is known from the DMA records that touch the same buffer).
  void set_sram_footprint(size_t b) { _sram_footprint_override = b; }
  size_t sram_footprint() const {
    return _sram_footprint_override ? _sram_footprint_override
                                    : _tile_numel * (_elem_bits / 8);
  }

  cycle_type finish_cycle = 0;
  cycle_type bubble_cycle=0;

  bool finished=false;
  int subgraph_id = 0;
 private:
  uint64_t _global_inst_id = 0;
  static uint64_t _next_global_inst_id;

  void *_owner = nullptr;
  std::list<std::shared_ptr<Instruction>>* _owner_ready_queue_ref = nullptr;
  bool* _owner_dirty_ref = nullptr;   // owning Core's _issue_dirty (re-arm gate)
  Opcode opcode;
  cycle_type compute_cycle = 0;
  cycle_type overlapping_cycle = 0;
  size_t ready_counter = 0;   // parents not yet finished; the minimal Instruction(Opcode)
                              // ctor (barriers) relies on this default + inc_ready_counter
  // Per-event subscriber sets: _deps[ISSUE] released at issue (occupancy),
  // _deps[DONE] released at finish (latency). std::set dedups + iterates in
  // instruction-id order (DepLess), so the release order does not depend on the
  // allocator.
  std::array<std::set<std::shared_ptr<Instruction>, DepLess>,
             static_cast<size_t>(DepEvent::COUNT)> _deps;
  std::vector<size_t> tile_size;
  std::vector<int> tile_stride;
  size_t _tile_numel = 0;
  size_t _nr_waiting_request=0;
  bool _got_first_response=false;
  size_t _elem_bits = 0;
  addr_type dram_addr = 0;
  uint32_t _numa_id = 0; // For DMA instruction
  int _compute_type = 0;
  std::vector<int64_t> _tag_idx_list;
  std::vector<int64_t> _tag_stride_list;
  std::vector<int64_t> _tag_key;
  std::vector<int64_t> _accum_tag_idx_list;
  std::vector<addr_type> _trace_address;
  std::string _addr_name;
  int64_t _addr_id = 0;
  int _nr_inner_loop = 0;
  bool _is_async_dma=false;
  bool _is_indirect_mode=false;
  bool _is_sparse_inst=false;
  std::string _indirect_index_path="";
  // SRAM-capacity model (see the setters above).
  int64_t _sram_alloc_id = -1;
  std::vector<int64_t> _sram_release_allocs;
  size_t _sram_footprint_override = 0;
  // SA weight-buffer model (see the setters above).
  int _assigned_sa = -1;
  std::shared_ptr<WeightToken> _weight_token;
  int _tile_group = -1;   // trace-only work-item id (see set_tile_group)
};