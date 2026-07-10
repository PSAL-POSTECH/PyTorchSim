#include "DMA.h"
#include "TileGraph.h"
#include "TraceLogTags.h"

DMA::DMA(uint32_t id, uint32_t dram_req_size, bool l2_datacache_enabled) {
  _id = id;
  _dram_req_size = dram_req_size;
  _l2_datacache_enabled = l2_datacache_enabled;
  _current_inst = nullptr;
  _finished = true;
}

void DMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  if (tile_size.size() <= 0 || tile_size.size() > get_max_dim()) {
    spdlog::error("[DMA {}] issued tile is not supported format.. tile.size: {}, tile_size: [{}]", _id, tile_size.size(), fmt::join(tile_size, ", "));
    exit(EXIT_FAILURE);
  }
  _finished = false;
}

std::shared_ptr<std::vector<mem_fetch*>> DMA::get_memory_access(cycle_type core_cycle, int nr_req) {

  if (!_generated_once) {
    std::shared_ptr<std::set<addr_type>> addr_set =
      _current_inst->get_dram_address(_dram_req_size);

    Tile* owner = (Tile*)_current_inst->get_owner();
    std::shared_ptr<TileSubGraph> owner_subgraph = owner->get_owner();
    unsigned long long base_daddr = _current_inst->get_base_dram_address();

    bool is_cacheable =
      owner_subgraph->is_cacheable(base_daddr, base_daddr + _dram_req_size);

    if (_l2_datacache_enabled && spdlog::should_log(spdlog::level::trace)) {
      spdlog::trace(
          "[{}][Core {}][{}][INST_ID={}] dram=0x{:016x} cacheable={}",
          core_cycle,
          _id,
          TraceLogTag::pad15(TraceLogTag::kL2CacheableStatusForAddress),
          _current_inst->get_global_inst_id(),
          base_daddr,
          is_cacheable);
    }
    if (spdlog::should_log(spdlog::level::trace))
    spdlog::trace(
        "[{}][Core {}][{}][INST_ID={}] core_id={} subgraph_id={} numa_id={} addr_name={} is_write={}",
        core_cycle,
        _id,
        TraceLogTag::pad15(TraceLogTag::kDmaNumaPlacement),
        _current_inst->get_global_inst_id(),
        owner_subgraph->get_core_id(),
        _current_inst->subgraph_id,
        _current_inst->get_numa_id(),
        _current_inst->get_addr_name(),
        _current_inst->is_dma_write());
    for (const auto& addr : *addr_set) {
      mem_access_type acc_type =
        _current_inst->is_dma_write() ? mem_access_type::GLOBAL_ACC_W
                                          : mem_access_type::GLOBAL_ACC_R;
      mf_type type =
        _current_inst->is_dma_write() ? mf_type::WRITE_REQUEST
                                          : mf_type::READ_REQUEST;

      mem_fetch* access = new mem_fetch(
          addr, acc_type, type, _dram_req_size,
          _current_inst->get_numa_id(),
          static_cast<void*>(_current_inst.get()));

      access->set_cacheable(is_cacheable);
      _current_inst->inc_waiting_request();
      _pending_accesses.push(access);
    }
    _generated_once = true;
  }

  if (nr_req == -1)
    nr_req = _pending_accesses.size();

  // Return pending accesses up to nr_req
  auto access_vec = std::make_shared<std::vector<mem_fetch *>>();
  for (int i = 0; i < nr_req; i++) {
      if (_pending_accesses.empty())
        break;
      access_vec->push_back(_pending_accesses.front());
      _pending_accesses.pop();
  }

  if (_pending_accesses.empty()) {
    _finished = true;
    _generated_once = false;
  }

  return access_vec;
}

uint32_t DMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}