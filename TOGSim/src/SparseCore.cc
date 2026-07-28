#include "SparseCore.h"
#include "TraceLogTags.h"

SparseCore::SparseCore(uint32_t id, SimulationConfig config) : Core(id, config) {
  /* Init stonne cores*/
  nr_cores = config.num_stonne_per_core;
  coreBusy.resize(nr_cores);
  traceCoreStatus.resize(nr_cores);
  traceCoreCycle.resize(nr_cores);
  traceNodeList.resize(nr_cores);
  traceLoadTraffic.resize(nr_cores);
  traceStoreTraffic.resize(nr_cores);
  percore_tiles.resize(nr_cores);
  stonneCores.resize(nr_cores);
  traceMode.resize(nr_cores);
  percore_stat.resize(nr_cores);
  percore_total_stat.resize(nr_cores);
  for (int i=0; i<nr_cores; i++) {
    SST_STONNE::sstStonne* core = new SST_STONNE::sstStonne(config.stonne_config_path);
    stonneCores.at(i) = core;
    stonneCores.at(i)->init(1);
    coreBusy.at(i) = false;
    traceCoreStatus.at(i) = 0;
    traceCoreCycle.at(i) = 0;
    percore_tiles.at(i) = std::vector<std::shared_ptr<Tile>>();
    percore_stat.at(i).reset();
    percore_total_stat.at(i).reset();
  }

  Config stonneConfig = stonneCores.at(0)->getStonneConfig();
  unsigned int core_freq_mhz = config.core_freq_mhz; // MHz;
  num_ms = stonneConfig.m_MSNetworkCfg.ms_size;
  r_port_nr = config.num_stonne_port;
  w_port_nr = config.num_stonne_port;

  double compute_throughput = static_cast<double>(num_ms) * core_freq_mhz / 1e3; // FLOPs/sec
  double dn_bandwidth = static_cast<double>(r_port_nr) * config.dram_req_size * core_freq_mhz * 1e6 / 8.0 / 1e9; // GB/s
  double rn_bandwidth = static_cast<double>(w_port_nr) * config.dram_req_size * core_freq_mhz * 1e6 / 8.0 / 1e9; // GB/s
  for (int i=0; i<nr_cores; i++) {
    spdlog::info("[Config/StonneCore {}][{}] Compute Throughput: {:.2f} GFLOPs/sec", id, i, compute_throughput);
    spdlog::info("[Config/StonneCore {}][{}] Distribution Network Bandwidth: {:.2f} GB/s",
                id, i, dn_bandwidth, r_port_nr);
    spdlog::info("[Config/StonneCore {}][{}] Reduction Network Bandwidth: {:.2f} GB/s",
                id, i, rn_bandwidth, w_port_nr);
  }
};

SparseCore::~SparseCore() {
  for (auto& core : stonneCores){
    delete core;
  }
}

bool SparseCore::running() {
  bool is_running = !_request_queue.empty() || !_response_queue.empty();
  for (auto& tile_vec : percore_tiles)
    is_running |= tile_vec.size();
  return is_running;
}

void SparseCore::issue(std::shared_ptr<Tile> tile) {
  int32_t selected_core_idx = -1;
  for (int i=0; i<nr_cores; i++) {
    int32_t core_idx = rr_idx % nr_cores;
    if (!coreBusy.at(i)) {
      selected_core_idx = core_idx;
      rr_idx = (selected_core_idx + 1) % nr_cores;
      break;
    }
  }
  if (selected_core_idx == -1) {
    spdlog::error("[StonneCore {}] Failed to issue tile", _id);
    exit(1);
  }
  stonneCores.at(selected_core_idx)->init(1);
  traceNodeList.at(selected_core_idx).clear();

  SST_STONNE::StonneOpDesc *opDesc = static_cast<SST_STONNE::StonneOpDesc*>(tile->get_custom_data());
  bool is_trace_mode = true;
  if (opDesc) {
    is_trace_mode = false;
    stonneCores.at(selected_core_idx)->setup(*opDesc, 0x1000000 * selected_core_idx); // FIXME. To avoid same address
    stonneCores.at(selected_core_idx)->init(1);
  }
  setTraceMode(selected_core_idx, is_trace_mode);
  percore_tiles.at(selected_core_idx).push_back(tile);
  coreBusy.at(selected_core_idx) = true;
  spdlog::info("[{}][StonneCore {}/{}][Launch] New operation (trace_mode: {})", _core_cycle, _id, selected_core_idx, is_trace_mode);
};

bool SparseCore::can_issue(const std::shared_ptr<Tile>& op) {
  bool idle_exist = false;
  for (bool flag : coreBusy) {
    idle_exist |= !flag;
  }
  return idle_exist && op->is_stonne_tile();
}

void SparseCore::checkStatus(uint32_t subcore_id) {
  auto& stonneCore = stonneCores.at(subcore_id);
  int new_status = stonneCore->getMCFSMStats();
  int compute_cycle = stonneCore->getMSStats().n_multiplications;
  if (traceCoreStatus.at(subcore_id) != new_status) {
    spdlog::trace("[{}][StonneCore {}/{}][Transition] status {} -> {}, Load/Store: {}/{}, compute_cycle: {}",
      _core_cycle, _id, subcore_id, traceCoreStatus.at(subcore_id), new_status,
      traceLoadTraffic.at(subcore_id).size(), traceStoreTraffic.at(subcore_id).size(), (compute_cycle - traceCoreCycle.at(subcore_id))/num_ms);
    if (traceLoadTraffic.at(subcore_id).size()) {
      TraceNode load_node = TraceNode(traceNodeList.at(subcore_id).size()+2, "load", TraceNode::StonneTraceLoad);
      load_node.setAddress(traceLoadTraffic.at(subcore_id));
      traceNodeList.at(subcore_id).push_back(load_node);
    }
    if (_core_cycle - traceCoreCycle.at(subcore_id)) {//((compute_cycle - traceCoreCycle.at(subcore_id))/num_ms) {
      TraceNode compute_node = TraceNode(traceNodeList.at(subcore_id).size()+2, "compute", TraceNode::StonneTraceCompute, _core_cycle - traceCoreCycle.at(subcore_id));
      traceNodeList.at(subcore_id).push_back(compute_node);
    }
    if (traceStoreTraffic.at(subcore_id).size()) {
      TraceNode store_node = TraceNode(traceNodeList.at(subcore_id).size()+2, "store", TraceNode::StonneTraceStore);
      store_node.setAddress(traceStoreTraffic.at(subcore_id));
      traceNodeList.at(subcore_id).push_back(store_node);
    }

    traceCoreStatus.at(subcore_id) = new_status;
    traceCoreCycle.at(subcore_id) = _core_cycle;
    traceLoadTraffic.at(subcore_id).clear();
    traceStoreTraffic.at(subcore_id).clear();
  }
}

void SparseCore::subCoreCycle(uint32_t subcore_id) {
  if (!traceMode.at(subcore_id)) {
    auto& stonneCore = stonneCores.at(subcore_id);
    stonneCore->cycle();

    /* Check FSM status transition */
    checkStatus(subcore_id);

    /* Send Memory Request */
    while (SimpleMem::Request* req = stonneCore->popRequest()) {
      uint64_t target_addr =  (req->getAddress() / _config.dram_req_size) * _config.dram_req_size;
      mem_access_type acc_type;
      mf_type type;

      switch(req->getcmd()) {
        case SimpleMem::Request::Read:
          acc_type = mem_access_type::GLOBAL_ACC_R;
          type = mf_type::READ_REQUEST;
          traceLoadTraffic.at(subcore_id).insert(target_addr);
          break;
        case SimpleMem::Request::Write:
          acc_type = mem_access_type::GLOBAL_ACC_W;
          type = mf_type::WRITE_REQUEST;
          traceStoreTraffic.at(subcore_id).insert(target_addr);
          break;
        default:
          spdlog::error("[StonneCore] Invalid request type from core");
          return;
      }
      req->request_time = _core_cycle;
      req->stonneId = subcore_id;
      std::tuple<uint64_t, mem_access_type, mf_type, int> key = std::make_tuple(target_addr, acc_type, type, allocTrafficID());
      registerMemfetch(key, [this, req, acc_type, type]() {
        spdlog::trace("[{}][StonneCore][DRAM Response] Round Trip Cycle: {}, Address: {:#x}, Request Type: {}, DRAM Req Size: {}", \
              _core_cycle, _core_cycle - req->request_time, req->getAddress(), int(req->getcmd()), _config.dram_req_size);
        req->setReply();
        stonneCores.at(req->stonneId)->pushResponse(req);
      });
    }

    /* Finish stonne core */
    if (coreBusy.at(subcore_id) && stonneCore->isFinished()) {
      stonneCore->finish();
      spdlog::info("[{}][StonneCore {}/{}][Finish] Operation done", _core_cycle, _id, subcore_id);
      std::shared_ptr<Tile> target_tile = percore_tiles.at(subcore_id).front();
      SST_STONNE::StonneOpDesc *opDesc = static_cast<SST_STONNE::StonneOpDesc*>(target_tile->get_custom_data());
      if (opDesc->trace_path != "")
        dumpTrace(subcore_id, opDesc->trace_path);

      target_tile->set_status(Tile::Status::FINISH);
      _finished_tiles.push(target_tile);
      percore_tiles.at(subcore_id).erase(percore_tiles.at(subcore_id).begin());
      coreBusy.at(subcore_id) = false;
    }
  } else {
    /* Check finished computation */
    auto& target_pipeline = get_compute_pipeline(0);
    if (!target_pipeline.empty()) {
      if (target_pipeline.front()->finish_cycle <= _core_cycle) {
        finish_instruction(target_pipeline.front());
        target_pipeline.pop();
      }
      percore_stat.at(subcore_id).n_multiplications += num_ms;
    }

    /* Check finished dma operation */
    bool retry=true;
    while (retry) {
      retry = false;
      for (auto it=_dma_finished_queue.begin();it!=_dma_finished_queue.end();it++) {
        std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
        /* Pass not finished instruction */
        if (instruction->get_waiting_request())
          continue;

        /* Finish DMA read instruction */
        if (instruction->is_dma_read())
          finish_instruction(instruction);
        /* Erase the instruction in DMA finished queue */
        _dma_finished_queue.erase(_dma_finished_queue.begin());
        retry = true;
        break;
      }
    }

    auto& tile_queue = percore_tiles.at(subcore_id);
    if (tile_queue.empty())
      return;
    auto& instructions = tile_queue.front()->get_instructions();

    /* Finish stonne core */
    if (coreBusy.at(subcore_id) && instructions.empty()) {
      std::shared_ptr<Tile> target_tile = percore_tiles.at(subcore_id).front();
      target_tile->set_status(Tile::Status::FINISH);
      _finished_tiles.push(target_tile);
      percore_tiles.at(subcore_id).erase(percore_tiles.at(subcore_id).begin());
      coreBusy.at(subcore_id) = false;
      return;
    }

    /* Peek instruction*/
    if (instructions.empty())
      return;
    auto& inst = instructions.front();
    if (!inst->is_ready())
      return;


    bool issued = false;
    switch (inst->get_opcode()) {
      case Opcode::MOVIN:
        {
          auto acc_type = mem_access_type::GLOBAL_ACC_R;
          auto type = mf_type::READ_REQUEST;
          spdlog::trace("[{}][StonneCore {}/{}][{}] {}",
                        _core_cycle,
                        _id,
                        subcore_id,
                        TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                        opcode_to_string(inst->get_opcode()));
          for (auto addr : inst->get_trace_address()) {
            addr = addr - (addr & _config.dram_req_size-1);
            inst->inc_waiting_request();
            std::tuple<uint64_t, mem_access_type, mf_type, int> key = std::make_tuple(addr, acc_type, type, allocTrafficID());
            uint64_t current_time = _core_cycle;
            registerMemfetch(key, [this, inst, addr, current_time, type]() {
              spdlog::trace("[{}][StonneCore {}][RESPONSE] Round Trip Cycle: {}, Address: {:#x}, Request Type: {}, DRAM Req Size: {}", \
                this->_core_cycle, _id, this->_core_cycle - current_time, addr, int(type), _config.dram_req_size);
              inst->dec_waiting_request();
            });
          }
          issued = true;
          _dma_finished_queue.push_back(std::move(inst));
        }
        break;
      case Opcode::MOVOUT:
        {
          auto acc_type = mem_access_type::GLOBAL_ACC_W;
          auto type = mf_type::WRITE_REQUEST;
          spdlog::trace("[{}][StonneCore {}/{}][{}] {}",
                        _core_cycle,
                        _id,
                        subcore_id,
                        TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                        opcode_to_string(inst->get_opcode()));
          for (auto addr : inst->get_trace_address()) {
            addr = addr - (addr & _config.dram_req_size-1);
            inst->inc_waiting_request();
            std::tuple<uint64_t, mem_access_type, mf_type, int> key = std::make_tuple(addr, acc_type, type, allocTrafficID());
            uint64_t current_time = _core_cycle;
            registerMemfetch(key, [this, inst, addr, current_time, type]() {
              spdlog::trace("[{}][StonneCore {}][RESPONSE] Round Trip Cycle: {}, Address: {:#x}, Request Type: {}, DRAM Req Size: {}", \
                this->_core_cycle, _id, this->_core_cycle - current_time, addr, int(type), _config.dram_req_size);
              inst->dec_waiting_request();
            });
          }
          issued = true;
          finish_instruction(inst);
          _dma_finished_queue.push_back(std::move(inst));
        }
        break;
      case Opcode::COMP:
        {
          auto& target_pipeline = get_compute_pipeline(0);
          if (target_pipeline.empty())
            inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
          else
            inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle();
          spdlog::trace("[{}][StonneCore {}/{}][{}] {}, finish_at={}",
                          _core_cycle,
                          _id,
                          subcore_id,
                          TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                          opcode_to_string(inst->get_opcode()),
                          inst->finish_cycle);
          target_pipeline.push(inst);
          issued = true;
        }
        break;
      default:
        spdlog::error("Undefined instruction opcode type");
        exit(EXIT_FAILURE);
    }
    if (issued) {
      instructions.erase(std::find(instructions.begin(), instructions.end(), inst));
    }
  }
}

void SparseCore::cycle() {
  _core_cycle++;
  /* Handle core cycle*/
  for (uint32_t subcore_id=0; subcore_id<stonneCores.size(); subcore_id++)
    subCoreCycle(subcore_id);

  /* Handle memory request/response */
  int nr_request = 0;
  while (!request_merge_table.empty() && nr_request <= r_port_nr) {
    for (auto& req_pair : request_merge_table) {
      _request_queue.push(req_pair.second);
      request_merge_table.erase(req_pair.first);
      spdlog::debug("[{}][StonneCore][{}] Address: {:#x}, Access Type: {}, Request Type: {}, DRAM Req Size: {}, nr_request: {}", \
              _core_cycle, _id, req_pair.second->get_addr(), int(req_pair.second->get_access_type()), int(req_pair.second->get_type()),
              _config.dram_req_size, nr_request);
      nr_request++;
      break;
    }
  }

  // Send Memory Response
  nr_request = 0;
  while (!_response_queue.empty()) {
    mem_fetch* resp_wrapper = _response_queue.front();
    auto* callbacks = static_cast<std::vector<std::function<void()>>*>(resp_wrapper->get_custom_data());
    for (int i=0; i<callbacks->size(); i++) {
      (*callbacks).at(i)();
    }
    delete callbacks;
    delete resp_wrapper;
    _response_queue.pop();
    if (++nr_request > w_port_nr)
      break;
  }

  /* Check print stat */
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0)
    print_current_stats();
}

bool SparseCore::has_memory_request() {
  return !_request_queue.empty();
}

void SparseCore::pop_memory_request() {
  _request_queue.pop();
}

void SparseCore::push_memory_response(mem_fetch* response) {
  _response_queue.push(response);
}

void SparseCore::print_current_stats() {
  spdlog::info("========= Sparse Core stat =========");
  for (size_t i = 0; i < stonneCores.size(); ++i) {
    if (!isTraceMode(i)) {
      MSwitchStats stats = stonneCores.at(i)->getMSStats();
      stats -= percore_total_stat.at(i);
      percore_stat.at(i) = stats;
      percore_total_stat.at(i) = stonneCores.at(i)->getMSStats();
    } else {
      percore_total_stat.at(i) += percore_stat.at(i);
    }
    cycle_type nr_mul = percore_stat.at(i).n_multiplications;
    percore_stat.at(i).reset();
    spdlog::info("StonneCore [{}][{}] : nr_multiplications: {}", _id, i, nr_mul);
  }
  spdlog::info("StonneCore [{}] : Total cycle {}", _id, _core_cycle);
}

void SparseCore::print_stats() {
  spdlog::info("========= Sparse Core stat =========");
  for (size_t i = 0; i < stonneCores.size(); ++i) {
    if (!isTraceMode(i)) {
      MSwitchStats stats = stonneCores.at(i)->getMSStats();
      stats -= percore_total_stat.at(i);
      percore_stat.at(i) = stats;
      percore_total_stat.at(i) = stats;
    } else {
      percore_total_stat.at(i) += percore_stat.at(i);
    }
    cycle_type nr_mul = percore_total_stat.at(i).n_multiplications;
    spdlog::info("StonneCore [{}][{}] : nr_multiplications: {}", _id, i, nr_mul);
  }
  spdlog::info("StonneCore [{}] : Total cycle {}", _id, _core_cycle);
}

std::shared_ptr<Tile> SparseCore::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void SparseCore::finish_instruction(std::shared_ptr<Instruction>& inst, InstFinishTraceTag tag) {
  if (tag == InstFinishTraceTag::DmaRespComplete) {
    if (!inst->finished) {
      spdlog::error("[{}][StonneCore {}][Error] ALL_DRAM_RESPONSES_RECEIVED trace but inst not finished",
                    _core_cycle,
                    _id);
      exit(EXIT_FAILURE);
    }
    spdlog::trace("[{}][StonneCore {}][{}][INST_ID={}] {}",
                    _core_cycle,
                    _id,
                    TraceLogTag::pad15(TraceLogTag::kAllDramResponsesReceived),
                    inst->get_global_inst_id(),
                    opcode_to_string(inst->get_opcode()));
    return;
  }
  if (inst->finished) {
    spdlog::error("[{}][StonneCore {}][Error] {} inst already finished!!", _core_cycle, _id,
                  opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  const char* trace_tag = (tag == InstFinishTraceTag::DmaIssueComplete)
                              ? TraceLogTag::kAsyncDmaAllRequestsIssued
                              : TraceLogTag::kInstructionFinished;
  const std::string tag15 = TraceLogTag::pad15(trace_tag);
  if (inst->get_opcode() == Opcode::COMP) {
    spdlog::info("[{}][StonneCore {}][{}] {}", _core_cycle, _id, tag15,
                 opcode_to_string(inst->get_opcode()));
  } else if (inst->get_opcode() == Opcode::MOVIN || inst->get_opcode() == Opcode::MOVOUT) {
    spdlog::info("[{}][StonneCore {}][{}] {}", _core_cycle, _id, tag15,
                 opcode_to_string(inst->get_opcode()));
  }
}

void SparseCore::registerMemfetch(const std::tuple<uint64_t, mem_access_type, mf_type, int>& key, std::function<void()> callback) {
  if (request_merge_table.find(key) == request_merge_table.end()) {
    mem_fetch* req_wrapper = new mem_fetch(std::get<0>(key), std::get<1>(key), std::get<2>(key), _config.dram_req_size, -1);

    auto* callbacks = new std::vector<std::function<void()>>();
    req_wrapper->set_custom_data(static_cast<void*>(callbacks));
    request_merge_table[key] = req_wrapper;
  }
  mem_fetch* req_wrapper = request_merge_table[key];
  auto* callbacks = static_cast<std::vector<std::function<void()>>*>(req_wrapper->get_custom_data());
  callbacks->push_back(callback);
}

void SparseCore::dumpTrace(int stonne_core_id, const std::string& path) {
  std::ofstream outFile(path);
  if (!outFile) {
    spdlog::error("[StonneCore] Failed to make trace dump file to \"{}\"", path);
    return;
  }
  // Static nodes for the graph
  outFile << "graph = {\n 0: {\n"
          << "    \"node_id\": 0,\n"
          << "    \"node_name\": \"root\",\n"
          << "    \"node_type\": 0,\n"
          << "    \"parents\": [],\n"
          << "    \"children\": [1]\n"
          << "  },\n"
          << "  1: {\n"
          << "    \"node_id\": 1,\n"
          << "    \"node_name\": \"loopNode\",\n"
          << "    \"node_type\": 2,\n"
          << "    \"parents\": [0],\n"
          << "    \"children\": [2],\n"
          << "    \"loop_index\": \"loop_arg000\",\n"
          << "    \"loop_start\": 0,\n"
          << "    \"loop_end\": 1,\n"
          << "    \"loop_step\": 1,\n"
          << "    \"loop_type\": \"outer_loop\""
          << "  },\n";

  // Output traceNodeList
  for (size_t i = 0; i < traceNodeList.at(stonne_core_id).size(); ++i) {
      if (i != 0) outFile << ",\n";
      outFile << traceNodeList.at(stonne_core_id)[i];
  }
  outFile << "\n}" << std::endl;
  spdlog::info("[{}][StonneCore] Success to save trace dump file to \"{}\"", _core_cycle, path);
}
