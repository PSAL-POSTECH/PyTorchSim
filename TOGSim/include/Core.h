#pragma once
#include <robin_hood.h>
#include <unordered_set>
#include <map>
#include <memory>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "Dram.h"
#include "Tile.h"
#include "SimulationConfig.h"
#include "DMA.h"
#include "TraceLogTags.h"

/** Log tag kind for Core::finish_instruction (see TraceLogTag names in TraceLogTags.h). */
enum class InstFinishTraceTag {
  Fnshed,
  DmaIssueComplete,
  DmaRespComplete,
};

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core()=default;
  virtual bool running();
  // True if this core has work actively in flight (DMA / compute pipeline / queues)
  // that will produce a future finish event -- i.e. running() minus "tiles waiting".
  // Used by the frozen-state (spad-too-small) guard.
  bool has_inflight();
  virtual bool can_issue(const std::shared_ptr<Tile>& op);
  virtual void issue(std::shared_ptr<Tile> tile);
  virtual std::shared_ptr<Tile> pop_finished_tile();
  virtual void cycle();
  virtual void print_stats();
  virtual void print_current_stats();
  virtual void finish_instruction(std::shared_ptr<Instruction>& inst,
                                  InstFinishTraceTag tag = InstFinishTraceTag::Fnshed);
  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual mem_fetch* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(mem_fetch* response);
  void check_tag() { _dma.check_table(); }
  void inc_numa_local_access() { _stat_numa_local_access++; }
  void inc_numa_remote_access() { _stat_numa_remote_access++; }

  std::queue<std::shared_ptr<Instruction>>& get_compute_pipeline(int compute_type);
  enum {
    VECTOR_UNIT,
    MATMUL,
    PRELOAD,
    NR_COMPUTE_UNIT
  };

 protected:
  void dma_cycle();
  void compute_cycle();
  void vu_cycle();
  void sa_cycle();
  bool can_issue_compute(std::shared_ptr<Instruction>& inst);
  void update_stats();
  // SRAM-capacity throttle (sec 10.4): a consumer frees the buffer-versions it
  // read (refcount -> 0 releases the spad bytes). Called when COMP/MOVOUT issue.
  void release_sram(const std::shared_ptr<Instruction>& inst);
  // SA weight-buffer throttle (sec 10.4): pick a systolic array that has a free
  // weight slot (round-robin among free); -1 if all full -> the preload stalls.
  int pick_free_weight_sa();
  // Free weight slots due this cycle: a matmul releases its slot at its
  // streaming-end (finish - overlapping, when it stops reading the weight),
  // scheduled at issue in _weight_release_q. Last consumer frees it.
  void process_weight_releases();

  /* Core id & config file */
  const uint32_t _id;
  const SimulationConfig _config;
  uint32_t _num_systolic_array_per_core;
  uint32_t _systolic_array_rr = 0;

  /* DMA Unit */
  DMA _dma;

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _stat_tot_vu_compute_cycle = 0;
  std::vector<cycle_type> _stat_tot_sa_compute_cycle;
  cycle_type _stat_tot_dma_cycle = 0;
  cycle_type _stat_tot_dma_idle_cycle = 0;
  cycle_type _stat_tot_vu_compute_idle_cycle = 0;
  std::vector<cycle_type> _stat_tot_sa_compute_idle_cycle;
  std::vector<uint64_t> _stat_inst_count;
  std::vector<uint64_t> _stat_tot_skipped_inst;
  uint64_t _stat_tot_mem_response = 0;
  uint64_t _stat_gemm_inst = 0;
  uint64_t _stat_skip_dma = 0;
  uint64_t _stat_numa_local_access = 0;
  uint64_t _stat_numa_remote_access = 0;

  cycle_type _stat_vu_compute_cycle = 0;
  std::vector<cycle_type> _stat_sa_compute_cycle;
  cycle_type _stat_dma_cycle = 0;
  cycle_type _stat_dma_idle_cycle = 0;
  cycle_type _stat_vu_compute_idle_cycle = 0;
  std::vector<cycle_type> _stat_sa_compute_idle_cycle;
  uint64_t _stat_mem_response = 0;

  std::vector<std::shared_ptr<Tile>> _tiles;
  std::queue<std::shared_ptr<Tile>> _finished_tiles;

  std::queue<std::shared_ptr<Instruction>> _vu_compute_pipeline;
  std::vector<std::queue<std::shared_ptr<Instruction>>> _sa_compute_pipeline;
  std::queue<std::shared_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::shared_ptr<Instruction>> _st_inst_queue;

  std::unordered_map<Instruction*, std::shared_ptr<Instruction>> _dma_waiting_queue;
  std::vector<std::shared_ptr<Instruction>> _dma_finished_queue;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
  uint32_t _waiting_write_reqs;

  // SRAM-capacity throttle (sec 10.4). _sram_used = current per-core spad bytes;
  // _sram_capacity = limit (0 = disabled); _sram_allocs maps a buffer-version id
  // to its accumulated footprint bytes (freed when its last reader issues).
  size_t _sram_used = 0;
  size_t _sram_capacity = 0;
  std::unordered_map<int64_t, size_t> _sram_allocs;

  // SA weight-buffer throttle (sec 10.4). _weight_slots_used[s] = weights resident
  // on SA s (loaded by a preload, not yet freed by their last matmul);
  // _weight_slot_depth = per-SA capacity (0 = disabled -> plain round-robin).
  std::vector<int> _weight_slots_used;
  uint32_t _weight_slot_depth = 0;
  // Pending weight-slot releases keyed by cycle (each matmul's streaming-end);
  // process_weight_releases() drains those due and decrements the token.
  std::multimap<cycle_type, std::shared_ptr<WeightToken>> _weight_release_q;
};