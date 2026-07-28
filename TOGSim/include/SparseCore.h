#include <map>
#include <vector>
#include <iomanip>
#include "Core.h"
#include "sstStonne.h"
#include "SimpleMem.h"
#include "Config.h"

class TraceNode {
private:
  int node_id;
  int node_type;
  std::string node_name;
  std::set<uint64_t> address_set;
  int compute_cycle;

public:
  enum TraceType {StonneTraceCompute=6, StonneTraceLoad=7, StonneTraceStore=8};
  TraceNode(int id, std::string name, int type, int cycle = 0)
      : node_id(id), node_name(name), node_type(type), compute_cycle(cycle) {}
  void setAddress(std::set<uint64_t> addr_set) { address_set = addr_set; }
  friend std::ostream& operator<<(std::ostream& os, const TraceNode& node) {
    os << "  " << node.node_id << ": {\n"
        << "    \"node_id\": " << node.node_id << ",\n"
        << "    \"node_name\": " << std::quoted(node.node_name) << ",\n"
        << "    \"node_type\": " << node.node_type << ",\n"
        << "    \"parents\": [0],\n"
        << "    \"trace_address\": [";

    bool first = true;
    for (uint64_t addr : node.address_set) {
      if (!first)
        os << ", ";
      os << addr;
      first = false;
    }

    os << "],\n"
        << "    \"trace_compute_cycle\": " << node.compute_cycle << "\n"
        << "  }";
    return os;
  }
};

class SparseCore : public Core {
public:
  SparseCore(uint32_t id, SimulationConfig config);
  ~SparseCore();
  bool running() override;
  bool can_issue(const std::shared_ptr<Tile>& op) override;
  void issue(std::shared_ptr<Tile> tile) override;
  void cycle() override;
  void subCoreCycle(uint32_t subcore_id);
  void stonneCycle(SST_STONNE::sstStonne *&stonneCore, uint32_t stonne_core_id, bool &retFlag);
  bool has_memory_request();
  void pop_memory_request();
  mem_fetch* top_memory_request() { return _request_queue.front(); }
  void push_memory_response(mem_fetch* response) override;
  void print_stats() override;
  void print_current_stats() override;
  std::shared_ptr<Tile> pop_finished_tile() override;
  void finish_instruction(std::shared_ptr<Instruction>& inst,
                          InstFinishTraceTag tag = InstFinishTraceTag::Fnshed) override;
  void dumpTrace(int stonne_core_id, const std::string& path);
  bool isTraceMode(int stonne_core_id) { return traceMode.at(stonne_core_id); }
  void setTraceMode(int stonne_core_id, bool mode) { traceMode.at(stonne_core_id) = mode; }
  void checkStatus(uint32_t subcore_id);
  void registerMemfetch(const std::tuple<uint64_t, mem_access_type, mf_type, int>& key, std::function<void()> callback);
  int allocTrafficID() { int id = traffic_id; traffic_id++; return 0; }
  uint32_t num_ms = 1;
  uint32_t r_port_nr = 1;
  uint32_t w_port_nr = 1;
  uint32_t nr_cores = 1;
private:
  uint32_t rr_idx = 0;
  std::vector<bool> coreBusy;
  std::vector<int> traceCoreStatus;
  std::vector<int> traceCoreCycle;
  std::vector<bool> traceMode;
  std::vector<std::vector<TraceNode>> traceNodeList;
  std::vector<std::set<uint64_t>> traceLoadTraffic; // To trace dma traffic
  std::vector<std::set<uint64_t>> traceStoreTraffic; // To trace dma traffic
  std::vector<std::vector<std::shared_ptr<Tile>>> percore_tiles;
  std::vector<SST_STONNE::sstStonne*> stonneCores;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
  std::map<std::tuple<uint64_t, mem_access_type, mf_type, int>, mem_fetch*> request_merge_table;
  std::vector<MSwitchStats> percore_stat;
  std::vector<MSwitchStats> percore_total_stat;
  int traffic_id=0;
};

