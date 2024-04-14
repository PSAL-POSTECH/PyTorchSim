#include "TMA.h"

TMA::TMA(uint32_t dram_req_size) {
  _dram_req_size = dram_req_size;
  _current_inst = nullptr;
}

void TMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  spdlog::info("[TMA] instruction issued ");
  _current_inst->print();
  if (tile_size.size() != 2) {
    spdlog::error("[TMA] issued tile is not [y,x] format..");
    exit(EXIT_FAILURE);
  }

  _tile_size_x = tile_size.at(1);
  _tile_size_y = tile_size.at(0);
  _tile_idx_stride = std::min(size_t(_dram_req_size / _current_inst->get_precision()), size_t(_tile_size_x));
}

MemoryAccess* TMA::get_memory_access() {
  if (_current_inst == nullptr)
    return nullptr;

  if (_current_inst->get_tile_numel() <= _tile_idx)
    return nullptr;

  addr_type addr = _current_inst->get_dram_address(_tile_idx / _tile_size_x, _tile_idx % _tile_size_x);
  MemoryAccess* access = new MemoryAccess({
    .id = generate_mem_access_id(),
    .dram_address = addr,
    .size = _dram_req_size,
    .write = _current_inst->is_dma_write(),
    .request = true,
    .owner_instruction = _current_inst.get()
  });

  /* Increase tile idx */
  _tile_idx += _tile_idx_stride;
  _current_inst->inc_waiting_request();
  if (_current_inst->get_tile_numel() <= _tile_idx)
    _finished = true;

  return access;
}

uint32_t TMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}