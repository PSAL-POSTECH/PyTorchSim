#include "Interconnect.h"

SimpleInterconnect::SimpleInterconnect(SimulationConfig config)
  :  _latency(config.icnt_latency) {
  _cycles = 0;
  _config = config;
  _n_nodes = config.num_cores * _config.icnt_injection_ports_per_core + config.dram_channels;
  _in_buffers.resize(_n_nodes);
  _out_buffers.resize(_n_nodes);
  _busy_node.resize(_n_nodes);
  _rr_next_src.resize(_n_nodes);
  for(int node = 0; node < _n_nodes; node++) {
    _busy_node[node] = false;
    _in_buffers.at(node).resize(_n_nodes);
    _rr_next_src[node] = 0;
  }
}


bool SimpleInterconnect::running() {
  return false;
}

void SimpleInterconnect::cycle() {
  for(int dest = 0; dest < _n_nodes; dest++) {
    int src_start = _rr_next_src[dest];
    bool pushed = false;

    for(int i = 0; i < _n_nodes; i++) {
      int src = (src_start + i) % _n_nodes;

      if (!_in_buffers[src][dest].empty() &&
          _in_buffers[src][dest].front().finish_cycle <= _cycles) {

        _out_buffers[dest].push(_in_buffers[src][dest].front().access);
        _in_buffers[src][dest].pop();
        _rr_next_src[dest] = (src + 1) % _n_nodes;
        pushed = true;
        break;
      }
    }
  }
  _cycles++;
}

void SimpleInterconnect::push(uint32_t src, uint32_t dest, mem_fetch* request) {
  SimpleInterconnect::Entity entity;
  if(_in_buffers[src][dest].empty())
    entity.finish_cycle =  _cycles + _latency;
  else
    entity.finish_cycle =  _in_buffers[src][dest].back().finish_cycle + 1;
  entity.dest = dest;
  entity.access = request;
  _in_buffers[src][dest].push(entity);
}

bool SimpleInterconnect::is_full(uint32_t nid, mem_fetch* request) {
  //TODO: limit buffersize
  return false;
}

bool SimpleInterconnect::is_empty(uint32_t nid) {
  return _out_buffers[nid].empty();
}

mem_fetch* SimpleInterconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return _out_buffers[nid].front();
}

void SimpleInterconnect::pop(uint32_t nid) {
  _out_buffers[nid].pop();
  // spdlog::trace("PUSH {}", _cycles);
}


Booksim2Interconnect::Booksim2Interconnect(SimulationConfig config) {
  _config = config;
  _n_nodes = config.num_cores * _config.icnt_injection_ports_per_core + config.dram_channels;
  spdlog::info("Initialize Booksim2");
  /* Same base as Simulator.cc DRAM: TORCHSIM_DIR + configs + path; absolute icnt_config_path replaces prefix. */
  char* onnxim_path_env = std::getenv("TORCHSIM_DIR");
  std::string onnxim_path =
      onnxim_path_env != NULL ? std::string(onnxim_path_env) : std::string("./");

  _config_path =
      fs::path(onnxim_path).append("configs").append(config.icnt_config_path).string();
  spdlog::info("Booksim 2 config path : {}", _config_path);
  print_config(_config_path);
  _booksim = std::make_unique<booksim2::Interconnect>(_config_path, _n_nodes);
  _ctrl_size = 8;
}

void Booksim2Interconnect::print_config(std::string config_path) {
  std::ifstream config_file(config_path);
  std::string line;
  spdlog::info("Booksim2 Configuration: ");
  while (std::getline(config_file, line)) {
    std::cout << line << std::endl;
  }
  config_file.close();
}

bool Booksim2Interconnect::running() {
  return false;
}

void Booksim2Interconnect::cycle() {
  _booksim->run();
}

void Booksim2Interconnect::push(uint32_t src, uint32_t dest, mem_fetch* request) {
  booksim2::Interconnect::Type type = get_booksim_type(request);
  uint32_t size = get_packet_size(request);
  _booksim->push(request, 0, 0, size, type, src, dest);
}

bool Booksim2Interconnect::is_full(uint32_t nid, mem_fetch* request) {
  uint32_t size = get_packet_size(request);
  return _booksim->is_full(nid, 0, size);
}

bool Booksim2Interconnect::is_empty(uint32_t nid) {
  return _booksim->is_empty(nid, 0);
}

mem_fetch* Booksim2Interconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return (mem_fetch*) _booksim->top(nid, 0);
}

void Booksim2Interconnect::pop(uint32_t nid) {
  assert(!is_empty(nid));
  _booksim->pop(nid, 0);
}

void Booksim2Interconnect::print_stats() {
  _booksim->print_stats();
}

booksim2::Interconnect::Type Booksim2Interconnect::get_booksim_type(mem_fetch* access) {
  booksim2::Interconnect::Type type;
  switch (access->get_type())
  {
  case mf_type::READ_REQUEST:
    type = booksim2::Interconnect::Type::READ;
    break;
  case mf_type::READ_REPLY:
    type = booksim2::Interconnect::Type::READ_REPLY;
    break;
  case mf_type::WRITE_REQUEST:
    type = booksim2::Interconnect::Type::WRITE;
    break;
  case mf_type::WRITE_ACK:
    type = booksim2::Interconnect::Type::WRITE_REPLY;
    break;
  default:
    spdlog::error("[Interconenct] Unexpected memory type...");
    break;
  }
  return type;
}

uint32_t Booksim2Interconnect::get_packet_size(mem_fetch* access) {
  uint32_t size;
  switch (access->get_type())
  {
  case mf_type::READ_REQUEST:
  case mf_type::WRITE_ACK:
    size = _ctrl_size;
    break;
  case mf_type::READ_REPLY:
  case mf_type::WRITE_REQUEST:
    size = access->get_data_size();
    break;
  default:
    spdlog::error("[Interconenct] Unexpected memory type...");
    break;
  }
  return size;
}