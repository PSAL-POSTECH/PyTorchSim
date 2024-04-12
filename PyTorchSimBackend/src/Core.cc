#include "Core.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_compute_cycle(0),
      _stat_idle_cycle(0),
      _stat_tma_cycle(0),
      _stat_issued_cycle(0),
      _compute_memory_stall_cycle(0),
      _tma(config.dram_req_size) {
  _waiting_write_reqs = 0;
  _sram_size = _config.sram_size;
  _remain_sram_size = _sram_size;
}

bool Core::can_issue(std::unique_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return op->required_sram_size + _remain_sram_size < _sram_size;
}

void Core::issue(std::unique_ptr<Tile> op) {
  _remain_sram_size += op->required_sram_size;
  _tiles.push_back(std::move(op));
}

std::unique_ptr<Tile> Core::pop_finished_tile() {
  std::unique_ptr<Tile> result = std::make_unique<Tile>(Tile{});
  result->status = Tile::Status::EMPTY;
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::compute_cycle() {
  if (!_compute_pipeline.empty()) {
    _stat_compute_cycle++;
    if(_compute_pipeline.front()->finish_cycle <= _core_cycle)
      _compute_pipeline.pop();
  }
}

void Core::dma_cycle() {
  if (_tma.is_finished()) {
    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::unique_ptr<Instruction>& inst = _ld_inst_queue.front();
      _tma.issue_tile(std::move(inst));
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::unique_ptr<Instruction>& inst = _st_inst_queue.front();
      _tma.issue_tile(std::move(inst));
      _st_inst_queue.pop();
    } else {
      /* TMA is idle */
      return;
    }
  }
  MemoryAccess *access = _tma.get_memory_access();
  /* Access couldn't be nullptr, since it is not finished */
  assert(access == nullptr);

  access->core_id = _id;
  access->start_cycle = _core_cycle;
  _request_queue.push(access);

  /* Increase tma stat cycle */
  _stat_tma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    std::unique_ptr<Instruction>& inst = _tiles[i]->instructions.front();
    /* Skip instruction is not ready */
    if (!inst->is_ready())
      continue;

    switch (inst->get_opcode()) {
      case Opcode::MOVIN:
        _ld_inst_queue.push(std::move(inst));
        issued = true;
        break;
      case Opcode::MOVOUT:
        _st_inst_queue.push(std::move(inst));
        issued = true;
        break;
      case Opcode::COMP:
        _ex_inst_queue.push(std::move(inst));
        issued = true;
        break;
      default:
        spdlog::error("Undefined instruction opcode type");
        exit(EXIT_FAILURE);
    }

    if (issued) {
      _tiles[i]->instructions.pop_front();
      if (_tiles[i]->instructions.empty()) {
        _tiles[i]->status = Tile::Status::FINISH;
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.pop_front();
      }
    }
  }

  /* Increate issue stall cycle */
  _stat_issued_cycle += (int)issued;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_compute_pipeline.empty();
  running = running || _waiting_write_reqs != 0;
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  running = running || !_ex_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() { return _request_queue.size() > 0; }

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(MemoryAccess *response) {
  if (response->write)
    _waiting_write_reqs--;
  delete response;
}

bool Core::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  spdlog::info("Core [{}] : Total tma {} Idle cycle {}", _id, _stat_tma_cycle, _stat_idle_cycle);
  spdlog::info("Core [{}] : Total cycle: {}", _id, _core_cycle);
}