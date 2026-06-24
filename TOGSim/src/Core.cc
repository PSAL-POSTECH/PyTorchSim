#include "Core.h"
#include "CoreTraceLog.h"
#include <spdlog/spdlog.h>
#include <algorithm>

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_dma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _dma(id, config.dram_req_size, config.l2d_type != L2CacheType::NOCACHE) {
  _sa_compute_pipeline.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _sram_capacity = (size_t)config.core_spad_size_kb * 1024;  // 0 = throttle disabled
  _weight_slot_depth = config.sa_weight_buffer_depth;        // 0 = disabled (plain rr)
  _weight_slots_used.resize(_num_systolic_array_per_core, 0);
}

// Round-robin a systolic array that still has a free weight slot; -1 if all full
// (the preload must stall). Advances _systolic_array_rr past the chosen SA.
int Core::pick_free_weight_sa() {
  for (uint32_t i = 0; i < _num_systolic_array_per_core; i++) {
    uint32_t s = (_systolic_array_rr + i) % _num_systolic_array_per_core;
    if (_weight_slots_used[s] < (int)_weight_slot_depth) {
      _systolic_array_rr = (s + 1) % _num_systolic_array_per_core;
      return (int)s;
    }
  }
  return -1;
}

void Core::process_weight_releases() {
  while (!_weight_release_q.empty() && _weight_release_q.begin()->first <= _core_cycle) {
    auto tok = _weight_release_q.begin()->second;
    _weight_release_q.erase(_weight_release_q.begin());
    if (--tok->refcount <= 0) _weight_slots_used[tok->sa]--;  // last reader frees the slot
  }
}

// The LAST reader of a buffer-version issued (bridge tags only that consumer):
// free the version's bytes back to the per-core spad.
void Core::release_sram(const std::shared_ptr<Instruction>& inst) {
  if (!_sram_capacity) return;
  for (int64_t id : inst->get_sram_release()) {
    auto it = _sram_allocs.find(id);
    if (it == _sram_allocs.end()) continue;
    _sram_used -= it->second;
    _sram_allocs.erase(it);
  }
}

bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return _tiles.size() < 4  && !op->is_stonne_tile();
}

void Core::issue(std::shared_ptr<Tile> op) {
  if (op->get_instructions().size()) {
    core_trace_log::trace_tile_scheduled(_core_cycle, _id,
                                         TraceLogTag::pad15(TraceLogTag::kTileScheduled));
  }
  for (const auto& inst : op->get_instructions()) {
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

std::queue<std::shared_ptr<Instruction>>& Core::get_compute_pipeline(int compute_type) {
  if (compute_type == VECTOR_UNIT)
    return _vu_compute_pipeline;
  else if (compute_type == MATMUL || compute_type == PRELOAD) {
    uint32_t sa_idx = _systolic_array_rr;
    _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
    return _sa_compute_pipeline.at(sa_idx);
  }
  else {
    spdlog::error("Undefined compute type");
    exit(EXIT_FAILURE);
  }
}

void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      _stat_vu_compute_cycle++;
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        cycle_type bubble = _vu_compute_pipeline.front()->bubble_cycle;
        _stat_vu_compute_idle_cycle += bubble;
        _stat_vu_compute_cycle = (bubble < _stat_vu_compute_cycle) ? (_stat_vu_compute_cycle - bubble) : 0;
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::sa_cycle() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_compute_pipeline.at(i).empty()) {
        if(_sa_compute_pipeline.at(i).front()->finish_cycle <= _core_cycle) {
          cycle_type bubble = _sa_compute_pipeline.at(i).front()->bubble_cycle;
          _stat_sa_compute_idle_cycle.at(i) += bubble;
          cycle_type& stat = _stat_sa_compute_cycle.at(i);
          stat = (bubble < stat) ? (stat - bubble) : 0;
          finish_instruction(_sa_compute_pipeline.at(i).front());
          _sa_compute_pipeline.at(i).pop();
        } else {
          _stat_sa_compute_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_compute_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
}

void Core::compute_cycle() {
  vu_cycle();
  sa_cycle();
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_dma.get_tag_finish(instruction->subgraph_id, key));
      spdlog::trace(
          "[{}][Core {}] TOG async DMA response (table notify): tag_addr=0x{:016x} global_inst_id={} "
          "subgraph_id={}",
          _core_cycle,
          _id,
          static_cast<uint64_t>(static_cast<uintptr_t>(instruction->get_addr_id())),
          instruction->get_global_inst_id(),
          instruction->subgraph_id);
      _dma.set_tag_finish(instruction->subgraph_id, key);
      finish_instruction(instruction, InstFinishTraceTag::DmaRespComplete);
      for (auto & wait_inst : _dma.get_tag_waiter(instruction->subgraph_id, key)) {
        _dma.mark_tag_used(instruction->subgraph_id, key);
        finish_instruction(wait_inst);
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  if (_dma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_dma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_dma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! */
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load; see TraceLogTag::kAsyncDmaAllRequestsIssued */
        finish_instruction(finished_inst, InstFinishTraceTag::DmaIssueComplete);
      } else if(!finished_inst->is_dma_read()) {
        core_trace_log::log_error_dma_instruction_invalid(_core_cycle, _id);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::MEMORY_BAR) {
        core_trace_log::trace_instruction_line(_core_cycle,
                                               _id,
                                               TraceLogTag::pad15(TraceLogTag::kInstructionFinished),
                                               finished_inst->get_global_inst_id(),
                                               core_trace_log::format_instruction_detail_line(
                                                   *finished_inst));
      }
      /*Pass to waiting queue */
      _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _ld_inst_queue.front();
      _dma.issue_tile(inst);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _st_inst_queue.front();
      _dma.issue_tile(inst);
      _st_inst_queue.pop();
    } else {
      /* DMA is idle */
      _stat_dma_idle_cycle++;
      return;
    }
  }
  /* Generate memfetch */
  auto access_vec = _dma.get_memory_access(_core_cycle, _config.icnt_injection_ports_per_core);
  for (auto access : *access_vec) {
    access->set_start_cycle(_core_cycle);
    _request_queue.push(access);
  }

  /* Increase dma stat cycle */
  _stat_dma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  process_weight_releases();  // free weight slots due this cycle before dispatch

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_ready_instructions();
    for (auto it=instructions.begin(); it!=instructions.end();) {
      auto& inst = *it;
      /* Skip instruction is not ready  */
      //if (!inst->is_ready())
      //  continue;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            /* Check another MOVIN with same tag is issued */
            auto& key = inst->get_tag_id();
            if (inst->is_sparse_inst()) {
              _dma.register_tag(inst->subgraph_id, key);
              _dma.set_tag_sparse(inst->subgraph_id, key);
              finish_instruction(inst);
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (inst->is_async_dma() && _dma.tag_key_exist(inst->subgraph_id, key)) {
              bool finished = _dma.get_tag_finish(inst->subgraph_id, key);
              if (finished)
                finish_instruction(inst);
              else
                _dma.register_tag_waiter(inst->subgraph_id, key, inst);
              core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionSkipped),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_dma_inst_issued_trace_line(
                                                           *inst));
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else {
              // SRAM-capacity gate: a load that would overflow the per-core spad does
              // not issue this cycle -- it stays in the ready queue until a consumer
              // frees a tile. On issue it occupies its buffer-version allocation.
              if (_sram_capacity && inst->get_sram_alloc() >= 0) {
                size_t F = inst->sram_footprint();
                // Stall if the tile does not fit the free spad now. If it can never
                // fit, Simulator::cycle() detects the frozen state and exits with a
                // "spad too small" error rather than looping forever.
                if (_sram_used + F > _sram_capacity)
                  break;                                       // not issued -> retry next cycle
                _sram_used += F;
                _sram_allocs[inst->get_sram_alloc()] += F;     // accumulate version footprint
              }
              core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionIssued),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_dma_inst_issued_trace_line(
                                                           *inst));
              _dma.register_tag(inst->subgraph_id, inst->get_tag_id());
              _ld_inst_queue.push(inst);
              issued = true;
              break;
            }
          }
        case Opcode::MOVOUT:
          release_sram(inst);   // store issued -> free the tiles it drained
          core_trace_log::trace_instruction_line(_core_cycle,
                                                   _id,
                                                   TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                                                   inst->get_global_inst_id(),
                                                   core_trace_log::format_dma_inst_issued_trace_line(
                                                       *inst));
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            const int ct = inst->get_compute_type();
            // SA selection + weight-buffer gate: a preload picks a systolic array with
            // a free weight slot and pins its matmul consumers to it (they free the slot
            // on finish). This bounds preload run-ahead and keeps matmuls on their SA.
            int sa_idx = -1;
            if (ct == MATMUL || ct == PRELOAD) {
              if (ct == PRELOAD) {
                int n_consumers = 0;   // matmuls reusing this weight
                for (auto& c : inst->get_pipeline_children())
                  if (c->get_compute_type() == MATMUL) n_consumers++;
                if (_weight_slot_depth > 0 && n_consumers > 0) {
                  sa_idx = pick_free_weight_sa();
                  if (sa_idx < 0) break;            // all weight slots full -> stall (retry)
                  _weight_slots_used[sa_idx]++;
                  auto tok = std::make_shared<WeightToken>(WeightToken{sa_idx, n_consumers});
                  for (auto& c : inst->get_pipeline_children())
                    if (c->get_compute_type() == MATMUL) {
                      c->set_assigned_sa(sa_idx);
                      c->set_weight_token(tok);
                    }
                } else {                            // disabled / no consumers -> plain rr
                  sa_idx = _systolic_array_rr;
                  _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
                }
              } else {                              // MATMUL
                sa_idx = inst->get_assigned_sa();
                if (sa_idx < 0) {                   // no preload pinned it -> rr fallback
                  sa_idx = _systolic_array_rr;
                  _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
                }
              }
              inst->set_assigned_sa(sa_idx);         // record the SA actually used (for the trace)
            }
            release_sram(inst);   // consumer issued -> free the tiles it read
            // sec 10.7: this op is now entering the pipeline -> release its
            // occupancy (pipeline) dependents so a preload/matmul successor
            // overlaps it instead of waiting its full latency.
            inst->release_pipeline_children();
            auto& target_pipeline = (ct == VECTOR_UNIT) ? _vu_compute_pipeline
                                                        : _sa_compute_pipeline.at(sa_idx);
            if (target_pipeline.empty()) {
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
              inst->bubble_cycle = inst->get_overlapping_cycle();
            } else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              int bubble_cycle = inst->get_overlapping_cycle() - overlapped_cycle;
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
              inst->bubble_cycle = bubble_cycle;
            }

            // Release this matmul's weight slot at its streaming-end (finish -
            // overlapping), not at full finish (the drain tail does not read it).
            if (ct == MATMUL && inst->get_weight_token()) {
              cycle_type rel = inst->finish_cycle > inst->get_overlapping_cycle()
                                 ? inst->finish_cycle - inst->get_overlapping_cycle() : _core_cycle;
              _weight_release_q.emplace(rel, inst->get_weight_token());
            }

            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              instructions.erase(it);
            } else {
              core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionIssued),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_instruction_detail_line(
                                                           *inst));
              target_pipeline.push(inst);
              issued = true;
              if (inst->get_compute_type()) {
                _stat_gemm_inst++;
              }
            }
          }
          break;
        case Opcode::MEMORY_BAR:
          {
            auto& key = inst->get_tag_id();
            uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
            if (finished == -1) {
              for (auto child_inst : inst->get_child_inst()) {
                if (child_inst->get_opcode() == Opcode::COMP && child_inst->get_compute_type() == MATMUL) {
                  child_inst->set_compute_cycle(0);
                }
              }
              finish_instruction(inst);
            } else if (finished != 0) {
              _dma.mark_tag_used(inst->subgraph_id, key);
              finish_instruction(inst);
            } else {
              _dma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            core_trace_log::trace_instruction_line(_core_cycle,
                                                     _id,
                                                     TraceLogTag::pad15(
                                                         TraceLogTag::kInstructionIssued),
                                                     inst->get_global_inst_id(),
                                                     core_trace_log::format_instruction_detail_line(
                                                         *inst));
            issued = true;
          }
          break;
        case Opcode::COMPUTE_BAR:
          {
            // Compute fence: finish only once ALL compute pipelines have drained
            // (every systolic array + the VPU empty). Until then it does not issue --
            // it stays in the ready queue and is re-checked next cycle.
            bool drained = _vu_compute_pipeline.empty();
            for (int s = 0; s < _num_systolic_array_per_core; s++)
              drained = drained && _sa_compute_pipeline.at(s).empty();
            if (drained) {
              core_trace_log::trace_instruction_line(_core_cycle, _id,
                  TraceLogTag::pad15(TraceLogTag::kInstructionFinished),
                  inst->get_global_inst_id(),
                  core_trace_log::format_instruction_detail_line(*inst));
              finish_instruction(inst);
              issued = true;
            }
          }
          break;
        default:
          core_trace_log::log_error_undefined_opcode();
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        instructions.erase(it);
        break;
      }
      it++;
    }
  }

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst, InstFinishTraceTag tag) {
  if (tag == InstFinishTraceTag::DmaRespComplete) {
    if (!inst->finished) {
      core_trace_log::log_error_dram_responses_trace_not_finished(_core_cycle, _id);
      exit(EXIT_FAILURE);
    }
    core_trace_log::trace_instruction_line(_core_cycle,
                                             _id,
                                             TraceLogTag::pad15(TraceLogTag::kAllDramResponsesReceived),
                                             inst->get_global_inst_id(),
                                             core_trace_log::format_instruction_detail_line(*inst));
    return;
  }
  if (inst->finished) {
    core_trace_log::log_error_instruction_already_finished(_core_cycle, _id,
                                                           opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  const char* trace_tag = (tag == InstFinishTraceTag::DmaIssueComplete)
                              ? TraceLogTag::kAsyncDmaAllRequestsIssued
                              : TraceLogTag::kInstructionFinished;
  core_trace_log::trace_instruction_line(_core_cycle,
                                           _id,
                                           TraceLogTag::pad15(trace_tag),
                                           inst->get_global_inst_id(),
                                           core_trace_log::format_instruction_detail_line(*inst));
}

bool Core::has_inflight() {
  // running() without the "_tiles.size() > 0" term: work that will produce a
  // finish event on its own (so the sim is NOT frozen). If this is false but
  // tiles remain, only stalled ready instructions are left.
  if (!_vu_compute_pipeline.empty()) return true;
  for (int i = 0; i < _num_systolic_array_per_core; i++)
    if (!_sa_compute_pipeline.at(i).empty()) return true;
  if (!_dma_waiting_queue.empty() || !_dma_finished_queue.empty()) return true;
  if (!_dma.empty()) return true;
  if (!_ld_inst_queue.empty() || !_st_inst_queue.empty()) return true;
  return false;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  for (int i=0; i<_num_systolic_array_per_core;i++)
    running = running || !_sa_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || !_dma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() {
  return !_request_queue.empty();
}

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_finished_queue.push_back(std::move(moved_inst));
      _dma_waiting_queue.erase(it);
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  std::vector<float> sa_utilization;
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i = 0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    auto opcode  = static_cast<Opcode>(i);
    auto inst = _stat_inst_count.at(i);
    auto skipped = _stat_tot_skipped_inst.at(i);
    auto name = opcode_to_string(opcode);

    if (opcode == Opcode::COMP) {
      auto gemm   = _stat_gemm_inst;
      auto vector = inst - gemm;
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {}), skipped inst_count {}",
            _id, name, inst, gemm, vector, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {})",
            _id, name, inst, gemm, vector);
    }
    else {
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {}, skipped inst_count: {}",
            _id, name, inst, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {}",
            _id, name, inst);
    }
  }
  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_tot_sa_compute_cycle.at(i) * 100) / _core_cycle);
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id, i, sa_utilization.at(i),
      _stat_tot_sa_compute_cycle.at(i), _stat_tot_sa_compute_idle_cycle.at(i));
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq_mhz / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response);
  spdlog::info("Core [{}] : Vector unit utilization(%): {:.2f}, active cycle: {}, idle_cycle: {}", _id,
    static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : NUMA local memory: {} requests, remote memory: {} requests", _id, _stat_numa_local_access, _stat_numa_remote_access);
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
}

void Core::print_current_stats() {
  std::vector<float> sa_utilization;
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_sa_compute_cycle.at(i) * 100) / _config.core_print_interval);
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq_mhz / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id, i, sa_utilization.at(i),
      _stat_sa_compute_cycle.at(i), _stat_sa_compute_idle_cycle.at(i));
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_dma_cycle, _stat_dma_idle_cycle, dram_bw, _stat_mem_response);
  spdlog::info("Core [{}] : Vector unit Utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id,
    static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    _stat_tot_sa_compute_cycle.at(i) += _stat_sa_compute_cycle.at(i);
    _stat_tot_sa_compute_idle_cycle.at(i) += _stat_sa_compute_idle_cycle.at(i);
    _stat_sa_compute_cycle.at(i) = 0;
    _stat_sa_compute_idle_cycle.at(i) = 0;
  }

  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_dma_cycle += _stat_dma_cycle;
  _stat_tot_dma_idle_cycle += _stat_dma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;

  _stat_vu_compute_cycle = 0;
  _stat_dma_cycle = 0;
  _stat_dma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
}