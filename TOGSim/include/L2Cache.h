#include <string>
#include <queue>
#include "Memfetch.h"
#include "Cache.h"
#include "Instruction.h"
#include "IntervalTree.h"

class L2CacheBase {
public:
  L2CacheBase(std::string name, CacheConfig &cache_config, uint32_t id, cycle_type *core_cycle,
    uint32_t l2d_hit_latency, std::queue<mem_fetch*> *to_xbar_queue,
    std::queue<mem_fetch*> *from_xbar_queue) : 
    l_name(name), l_cache_config(cache_config), l_id(id), l_core_cycle(core_cycle),
    l2d_hit_latency(l2d_hit_latency),
    l_to_xbar_queue(to_xbar_queue), l_from_xbar_queue(from_xbar_queue) {}
  virtual void cycle()=0;
  // Push memory response from DRAM
  virtual bool push(mem_fetch* req)=0;
  // Pop memory request from Cache
  void pop() { l_to_mem_queue.pop(); }
  mem_fetch* top() { return l_to_mem_queue.empty() ? NULL : l_to_mem_queue.front(); }
  virtual void print_stats() {};

protected:
  cycle_type *l_core_cycle;   // Core cycle
  std::string l_name;         // L2 name
  CacheConfig l_cache_config; // L2 cache config
  uint32_t l_id;              // L2 partition id
  uint32_t l2d_hit_latency;
  uint32_t n_read_port = 2; // Number of read ports (CMEM Read 2TB/s)
  uint32_t n_write_port = 1; // Number of write ports (CMEM Write 1TB/s)
  std::vector<uint32_t> read_port; // Current read port
  std::vector<uint32_t> write_port; // Current write port
  std::queue<mem_fetch*> *l_to_xbar_queue;
  std::queue<mem_fetch*> *l_from_xbar_queue;
  std::queue<mem_fetch*> l_to_mem_queue;
  DelayQueue<mem_fetch*> l_from_cache_queue;
  std::unique_ptr<Cache> l_cache;
};

class NoL2Cache : public L2CacheBase {
public:
  NoL2Cache(std::string name,  CacheConfig &cache_config, uint32_t id, cycle_type *core_cycle,
    std::queue<mem_fetch*> *to_xbar_queue, std::queue<mem_fetch*> *from_xbar_queue) : 
    L2CacheBase(name, cache_config, id, core_cycle, 0, to_xbar_queue, from_xbar_queue) {}
  void cycle() override;
  bool push(mem_fetch* req) override;  // Push memory response from DRAM
};

class L2DataCache : public L2CacheBase {
public:
  typedef IntervalTree<new_addr_type, int> CachePlan;
  L2DataCache(std::string name,  CacheConfig &cache_config, uint32_t id, cycle_type *core_cycle,
    uint32_t l2d_hit_latency, uint32_t num_cores, std::queue<mem_fetch*> *to_xbar_queue,
    std::queue<mem_fetch*> *from_xbar_queue);
  void cycle() override;
  bool push(mem_fetch* req) override;  // Push memory response from DRAM
  bool port_free(mem_fetch* req);
  void clear_port() { std::fill(read_port.begin(), read_port.end(), 0);
                      std::fill(write_port.begin(), write_port.end(), 0); }
  virtual void print_stats() override;
private:
  uint32_t _n_cores;
};