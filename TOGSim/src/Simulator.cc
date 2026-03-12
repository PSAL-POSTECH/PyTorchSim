#include "Simulator.h"

Simulator::Simulator(SimulationConfig config)
    : _config(config), _core_cycles(0) {
  // Create dram object
  _core_period = 1000000 / (config.core_freq_mhz);
  _icnt_period = 1000000 / (config.icnt_freq_mhz);
  _dram_period = 1000000 / (config.dram_freq_mhz);
  _core_time = 0;
  _dram_time = 0;
  _icnt_time = 0;
  _slot_id = 0;
  _max_slot = 2;
  _n_cores = config.num_cores;
  _n_memories = config.dram_channels;
  _memory_req_size = config.dram_req_size;
  _noc_node_per_core = config.icnt_injection_ports_per_core;
  char* onnxim_path_env = std::getenv("TORCHSIM_DIR");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env): std::string("./");

  // Create core objects
  _cores.resize(_n_cores);
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    if (config.core_type[core_index] == CoreType::WS_MESH) {
      spdlog::info("[Config/Core] Core {}: {} MHz, Systolic array per core: {}",
        core_index, config.core_freq_mhz, config.num_systolic_array_per_core);
      _cores.at(core_index) = std::make_unique<Core>(core_index, _config);
    } else if(config.core_type[core_index] == CoreType::STONNE) {
      spdlog::info("[Config/Core] Core {}: {} MHz, Stonne Core selected", core_index, config.core_freq_mhz);
      _cores.at(core_index) = std::make_unique<SparseCore>(core_index, _config);
    } else {
      throw std::runtime_error(fmt::format("Not implemented Core type {} ",
                                          (int)config.core_type[core_index]));
    }
  }

  if (config.dram_type == DramType::SIMPLE) {
    _dram = std::make_unique<SimpleDRAM>(config, &_core_cycles);
  } else if (config.dram_type == DramType::RAMULATOR2) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("[Config/DRAM] Ramulator2 config path: {}", ramulator_config);
    YAML::Node dram_config = YAML::LoadFile(ramulator_config);
    spdlog::info("Ramulator2 config: ");
    std::cout << dram_config << std::endl;
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator2>(config, &_core_cycles);
  } else {
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  }

  // Create interconnect object
  spdlog::info("[Config/Interconnect] Interconnect freq: {} MHz", config.icnt_freq_mhz);
  if (config.icnt_type == IcntType::SIMPLE) {
    spdlog::info("[Config/Interconnect] SimpleInerconnect selected");
    _icnt = std::make_unique<SimpleInterconnect>(config);
  } else if (config.icnt_type == IcntType::BOOKSIM2) {
    spdlog::info("[Config/Interconnect] BookSim2 selected");
    _icnt = std::make_unique<Booksim2Interconnect>(config);
  } else {
    spdlog::error("[Configuration] Invalid interconnect type...!");
    exit(EXIT_FAILURE);
  }
  _icnt_interval = config.icnt_stats_print_period_cycles;

  // Initialize Scheduler
  for (int i=0; i<config.num_partition;i++)
    _partition_scheduler.push_back(std::make_unique<Scheduler>(Scheduler(config, &_core_cycles, &_core_time, i)));
}

void Simulator::run_simulator() {
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::core_cycle() {
  for (int i=0; i<_max_slot; i++, _slot_id=(_slot_id + 1) % _max_slot) {
    // Issue new tile to core
    for (int core_id = 0; core_id < _n_cores; core_id++) {
      const std::shared_ptr<Tile> tile = get_partition_scheduler(core_id)->peek_tile(core_id, _slot_id, _config.core_type[core_id]);
      if (tile->get_status() != Tile::Status::EMPTY && _cores[core_id]->can_issue(tile))  {
        if (tile->get_status() == Tile::Status::INITIALIZED) {
          _cores[core_id]->issue(std::move(get_partition_scheduler(core_id)->get_tile(core_id, _slot_id)));
          break;
        } else {
          spdlog::error("[Simulator] issued tile is not valid status...!");
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  for (int core_id = 0; core_id < _n_cores; core_id++) {
      std::shared_ptr<Tile> finished_tile = _cores[core_id]->pop_finished_tile();
      if (finished_tile->get_status() == Tile::Status::FINISH) {
        get_partition_scheduler(core_id)->finish_tile(std::move(finished_tile));
      }
    _cores[core_id]->cycle();
  }
  /* L2 cache */
  _dram->cache_cycle();
  _core_cycles++;
}

void Simulator::dram_cycle() {
  _dram->cycle();
}

void Simulator::icnt_cycle() {
  _icnt_cycle++;

  for (int core_id = 0; core_id < _n_cores; core_id++) {
    for (int noc_id = 0; noc_id < _noc_node_per_core; noc_id++) {
    // PUHS core to ICNT. memory request
      int port_id = core_id * _noc_node_per_core + noc_id;
      if (_cores[core_id]->has_memory_request()) {
        mem_fetch *front = _cores[core_id]->top_memory_request();
        front->set_core_id(core_id);
        if (!_icnt->is_full(port_id, front)) {
          int node_id = _dram->get_channel_id(front) / _config.dram_channels_per_partitions;
          if (get_partition_id(core_id) == node_id)
            _cores[core_id]->inc_numa_local_access();
          else
            _cores[core_id]->inc_numa_remote_access();
          _icnt->push(port_id , get_dest_node(front), front);
          _cores[core_id]->pop_memory_request();
          _nr_from_core++;
        }
      }
      // Push response from ICNT. to Core.
      if (!_icnt->is_empty(port_id)) {
        _cores[core_id]->push_memory_response(_icnt->top(port_id));
        _icnt->pop(port_id);
        _nr_to_core++;
      }
    }
  }

  for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
    // ICNT to memory
    int core_offset = _n_cores * _noc_node_per_core;
    if (!_icnt->is_empty(core_offset + mem_id) &&
        !_dram->is_full(mem_id, _icnt->top(core_offset + mem_id))) {
      _dram->push(mem_id, _icnt->top(core_offset + mem_id));
      _icnt->pop(core_offset + mem_id);
      _nr_to_mem++;
    }
    // Pop response to ICNT from dram
    if (!_dram->is_empty(mem_id) &&
        !_icnt->is_full(core_offset + mem_id, _dram->top(mem_id))) {
      _icnt->push(core_offset + mem_id, get_dest_node(_dram->top(mem_id)),
                  _dram->top(mem_id));
      _dram->pop(mem_id);
      _nr_from_mem++;
    }
  }
  if (_icnt_interval!=0 && _icnt_cycle % _icnt_interval == 0) {
    spdlog::info("[ICNT] Core->ICNT request {}GB/Sec", ((_memory_req_size*_nr_from_core*(1000/_icnt_period)/_icnt_interval)));
    spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec", ((_memory_req_size*_nr_to_core*(1000/_icnt_period)/_icnt_interval)));
    spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec", ((_memory_req_size*_nr_to_mem*(1000/_icnt_period)/_icnt_interval)));
    spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec", ((_memory_req_size*_nr_from_mem*(1000/_icnt_period)/_icnt_interval)));
    _nr_from_core=0;
    _nr_to_core=0;
    _nr_to_mem=0;
    _nr_from_mem=0;
  }
  _icnt->cycle();
}

void Simulator::cycle() {
  while (running() || _core_cycles < 1) {
    set_cycle_mask();
    // Core Cycle
    if (IS_CORE_CYCLE(_cycle_mask))
      core_cycle();

    // DRAM cycle
    if (IS_DRAM_CYCLE(_cycle_mask))
      dram_cycle();

    // Interconnect cycle
    if (IS_ICNT_CYCLE(_cycle_mask))
      icnt_cycle();
  }
  for (auto &core: _cores) {
    core->check_tag();
  }
}

bool Simulator::running() {
  bool running = false;
  for (auto &core : _cores) {
    running = running || core->running();
  }
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    running = running || !get_partition_scheduler(core_id)->empty(core_id);
  }
  running = running || _icnt->running();
  running = running || _dram->running();
  return running;
}

void Simulator::set_cycle_mask() {
  _cycle_mask = 0x0;
  uint64_t minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_dram_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _dram_time += _dram_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
}

uint32_t Simulator::get_dest_node(mem_fetch *access) {
  switch (access->get_type())
  {
  case mf_type::READ_REQUEST:
  case mf_type::WRITE_REQUEST:
    return _config.num_cores * _noc_node_per_core + _dram->get_channel_id(access);
    break;
  case mf_type::READ_REPLY:
  case mf_type::WRITE_ACK:
    return access->get_core_id() * _noc_node_per_core + (_dram->get_channel_id(access) % _noc_node_per_core);
    break;
  default:
    spdlog::error("Unexpected memfetc type...");
    return -1;
    break;
  }
}

void Simulator::print_core_stat()
{
  _icnt->print_stats();
  _dram->print_stat();
  _dram->print_cache_stats();
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    _cores[core_id]->print_stats();
  }
  spdlog::info("Total execution cycles: {}", _core_cycles);
}
