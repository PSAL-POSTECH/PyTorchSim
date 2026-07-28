#include "L2Cache.h"

bool NoL2Cache::push(mem_fetch* req) {
  l_to_xbar_queue->push(req);
  return true;
}
void NoL2Cache::cycle() {
  if (!l_from_xbar_queue->empty()) {
    mem_fetch* req = l_from_xbar_queue->front();
    l_to_mem_queue.push(req);
    l_from_xbar_queue->pop();
  }
}

L2DataCache::L2DataCache(std::string name,  CacheConfig &cache_config, uint32_t id,
  cycle_type *core_cycle, uint32_t l2d_hit_latency, uint32_t num_cores,
  std::queue<mem_fetch*> *to_xbar_queue, std::queue<mem_fetch*> *from_xbar_queue) :
  L2CacheBase(name, cache_config, id, core_cycle, l2d_hit_latency, to_xbar_queue, from_xbar_queue), _n_cores(num_cores) {
  l_cache = std::make_unique<DataCache>(name, cache_config, id, 0, &l_to_mem_queue);
  l_from_cache_queue = DelayQueue<mem_fetch*>(l_name + "_latency_queue", true, 0);
  read_port.resize(num_cores, 0);
  write_port.resize(num_cores, 0);
}

bool L2DataCache::push(mem_fetch* req) {
  bool is_cacheable = req->is_cacheable();
  if (!is_cacheable) {
    l_to_xbar_queue->push(req);
  } else {
    if (l_cache->waiting_for_fill(req)) {
      if (!l_cache->fill_port_free())
        return false;
      l_cache->fill(req, *l_core_cycle);
    } else {
      if (req->get_access_type() == GLOBAL_ACC_R || req->get_access_type() == GLOBAL_ACC_W)
        l_to_xbar_queue->push(req);
    }
  }
  return true;
}

void L2DataCache::cycle() {
  l_from_cache_queue.cycle();
  l_cache->cycle();

  // Mem to Cache
  uint32_t line_size = l_cache_config.get_line_size();
  uint32_t sector_size = l_cache_config.get_sector_size();
  clear_port();

  /* Pass a request to cache */
  for (int i = 0; i < (n_read_port + n_write_port) * _n_cores; i++) {
    if (!l_from_xbar_queue->empty()) {
      mem_fetch* req = l_from_xbar_queue->front();
      /* Check cache plan */
      bool is_cacheable = req->is_cacheable();

      /* Go to l2 cache */
      if (is_cacheable && l_cache->data_port_free()) {
        if (!port_free(req)) continue;
        req->set_access_sector_mask(line_size, sector_size);
        std::deque<CacheEvent> events;
        CacheRequestStatus status = l_cache->access(
            req->get_addr(), *l_core_cycle, req, events);
        bool write_sent = CacheEvent::was_write_sent(events);
        bool read_sent = CacheEvent::was_read_sent(events);
        if (status == HIT) {
          if (!write_sent) {
            req->set_reply();
            req->current_state = "L2HIT";
            l_from_cache_queue.push(req, l2d_hit_latency);
          }
          l_from_xbar_queue->pop();
        } else if (status != RESERVATION_FAIL) {
          req->current_state = "L2MISS";
          if (req->is_write() &&
              (l_cache_config.get_write_alloc_policy() == FETCH_ON_WRITE ||
                l_cache_config.get_write_alloc_policy() == LAZY_FETCH_ON_READ)) {
            req->set_reply();
            req->current_state = "L2MISS-WRITE";
            l_from_cache_queue.push(req, l2d_hit_latency);
          }
          l_from_xbar_queue->pop();
        } else {
          // Status Reservation fail, Retry it
          assert(!write_sent);
          assert(!read_sent);
        }
      } else if (!is_cacheable) {
        l_to_mem_queue.push(req);
        l_from_xbar_queue->pop();
      }
    }

    if (l_cache->access_ready() &&
        !l_from_cache_queue.full()) {
      mem_fetch* req = l_cache->top_next_access();
      if (req->is_request()) req->set_reply();
      l_from_cache_queue.push(req, l2d_hit_latency);
      l_cache->pop_next_access();
    }

    if (l_from_cache_queue.arrived()) {
      mem_fetch* req = l_from_cache_queue.top();
      if (req->get_access_type() == GLOBAL_ACC_R || req->get_access_type() == GLOBAL_ACC_W)
        l_to_xbar_queue->push(req);
      l_from_cache_queue.pop();
    }
  }
}

bool L2DataCache::port_free(mem_fetch* req) {
  int core_id = req->get_core_id();
  if (req->is_write()) {
    write_port[core_id]++;
    if (write_port[core_id] > n_write_port) {
      return false; // No more write port available
    }
  } else {
    read_port[core_id]++;
    if (read_port[core_id] > n_read_port) {
      return false; // No more read port available
    }
  }
  return true; // Port is free
}

void L2DataCache::print_stats() {
  if (l_id == 0) {
    l_cache->get_stats().print_stats(stdout, l_name.c_str());
  }
}