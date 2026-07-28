#ifndef CACHE_H_
#define CACHE_H_
#include <bitset>
#include <cassert>
#include <cstdint>
#include <list>
#include <queue>
#include <memory>
#include <string>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "Cache_defs.h"
#include "Cache_stats.h"
#include "DelayQueue.h"
#include "Memfetch.h"

class CacheConfig {
 public:
  CacheConfig() {}
  void init(std::string config);
  bool disabled() const { return m_disabled; }
  uint32_t get_line_size() const { return m_line_size; }
  uint32_t get_atom_size() const { return m_atom_size; }
  uint32_t get_num_lines() const { return m_nset * m_assoc; }
  uint32_t get_num_assoc() const { return m_assoc; }
  uint32_t get_max_assoc() const { return m_origin_assoc; }
  uint32_t get_max_sets() const { return m_origin_nset; }
  uint32_t get_num_sets() const { return m_nset; }
  void set_sets(uint32_t sets) { m_nset = sets; }
  void set_assoc (uint32_t assoc) { m_assoc = assoc; }
  uint32_t get_mshr_entries() const { return m_mshr_entries; }
  uint32_t get_mshr_max_merge() const { return m_mshr_max_merge; }
  uint32_t get_miss_queue_size() const { return m_miss_queue_size; }
  uint32_t get_sector_size() { return m_sector_size; }
  uint32_t get_set_index(uint64_t addr) const;
  uint64_t get_tag(uint64_t addr) const;
  uint64_t get_block_addr(uint64_t addr) const;
  uint64_t get_mshr_addr(uint64_t addr) const;
  CacheType get_cache_type() const { return m_cache_type; }
  EvictPolicy get_evict_policy() const { return m_evict_policy; }
  WritePolicy get_write_policy() const { return m_write_policy; }
  WriteAllocatePolicy get_write_alloc_policy() const {
    return m_write_alloc_policy;
  }
  AllocationPolicy get_alloc_policy() const { return m_alloc_policy; }
  MshrConfig get_mshr_config() const { return m_mshr_type; }
  uint32_t get_nset() const { return m_nset; }
  uint32_t get_total_size_in_kb() const {
    return (m_line_size * m_nset * m_assoc) / 1024;
  }
  uint32_t get_origin_size () const {
    return m_line_size * m_origin_assoc * m_origin_nset;
  }
  uint32_t get_data_port_width() const { return m_data_port_width; }
 protected:
  bool m_valid = false;
  bool m_disabled = false;
  uint32_t m_origin_nset = 0;
  uint32_t m_line_size = 0;
  uint32_t m_line_size_log2 = 0;
  uint32_t m_nset = 0;
  uint32_t m_nset_log2 = 0;
  uint32_t m_assoc = 0;
  uint32_t m_origin_assoc = 0;
  uint32_t m_atom_size = 0;
  uint32_t m_sector_size = 0;
  uint32_t m_sector_size_log2 = 0;
  uint32_t m_mshr_entries = 0;
  uint32_t m_mshr_max_merge = 0;
  uint32_t m_miss_queue_size = 0;
  uint32_t m_result_fifo_entries = 0;
  uint32_t m_data_port_width = 0;
  CacheType m_cache_type;
  EvictPolicy m_evict_policy;
  WritePolicy m_write_policy;
  WriteAllocatePolicy m_write_alloc_policy;
  AllocationPolicy m_alloc_policy;
  MshrConfig m_mshr_type;
  SetIndexFunction m_set_index_function;

  uint32_t hash_function(uint64_t addr) const;
};

class CacheBlock {
 public:
  virtual void allocate(uint64_t tage, uint64_t block_addr, uint32_t time,
                        SectorMask sector_mask) = 0;
  virtual void fill(uint32_t time, SectorMask sector_mask) = 0;
  virtual bool match_tag(uint64_t tag) { return m_tag == tag; }
  virtual uint64_t get_block_addr() { return m_block_addr; }
  virtual bool is_valid_line() = 0;
  virtual bool is_invalid_line() = 0;
  virtual bool is_reserved_line() = 0;
  virtual bool is_modified_line() = 0;
  virtual SectorMask get_dirty_mask() = 0;
  virtual CacheBlockState get_status(SectorMask mask) = 0;
  virtual void set_status(CacheBlockState status, SectorMask mask) = 0;
  virtual bool is_readable(SectorMask mask) = 0;
  virtual uint64_t get_last_access_time() = 0;
  virtual uint64_t get_alloc_time() = 0;
  virtual void set_ignore_on_fill(bool ignore, SectorMask sector_mask) = 0;
  virtual void set_modified_on_fill(bool modified, SectorMask sector_mask) = 0;
  virtual void set_readable(bool readable, SectorMask sector_mask) = 0;
  virtual void set_last_access_time(uint64_t time, SectorMask sector_mask) = 0;
  virtual uint32_t get_modified_size() = 0;

 protected:
  uint64_t m_tag;
  uint64_t m_block_addr;
};

class LineCacheBlock : public CacheBlock {
 public:
  LineCacheBlock(uint32_t sector_size) : m_sector_size(sector_size) {};
  virtual void allocate(uint64_t tag, uint64_t block_addr, uint32_t time,
                        SectorMask sector_mask) override;
  virtual void fill(uint32_t time, SectorMask sector_mask) override;
  virtual bool is_valid_line() override { return m_status == VALID; }
  virtual bool is_invalid_line() override { return m_status == INVALID; }
  virtual bool is_reserved_line() override { return m_status == RESERVED; }
  virtual bool is_modified_line() override { return m_status == MODIFIED; }
  virtual SectorMask get_dirty_mask() override;
  virtual CacheBlockState get_status(SectorMask mask) override {
    return m_status;
  }
  virtual void set_status(CacheBlockState status, SectorMask mask) override {
    m_status = status;
  }
  virtual bool is_readable(SectorMask mask) override { return m_readable; }
  virtual uint64_t get_last_access_time() override {
    return m_last_access_time;
  }
  virtual uint64_t get_alloc_time() override { return m_alloc_time; }
  virtual void set_ignore_on_fill(bool ignore,
                                  SectorMask sector_mask) override {
    m_ignore_on_fill_status = ignore;
  }
  virtual void set_modified_on_fill(bool modified,
                                    SectorMask sector_mask) override {
    m_set_modified_on_fill = modified;
  }
  virtual void set_readable(bool readable, SectorMask sector_mask) override {
    m_readable = readable;
  }
  virtual void set_last_access_time(uint64_t time,
                                    SectorMask sector_mask) override {
    m_last_access_time = time;
  }
  virtual uint32_t get_modified_size() override {
    return SECTOR_CHUNCK_SIZE * m_sector_size;
  }

 protected:
  uint64_t m_alloc_time = 0;
  uint32_t m_sector_size = 0;
  uint64_t m_last_access_time = 0;
  uint64_t m_fill_time = 0;
  CacheBlockState m_status = INVALID;
  bool m_ignore_on_fill_status = false;
  bool m_set_modified_on_fill = false;
  bool m_readable = true;
};

class SectorCacheBlock : public CacheBlock {
 public:
  SectorCacheBlock(uint32_t sector_size) : m_sector_size(sector_size) {};
  virtual void allocate(uint64_t tag, uint64_t block_addr, uint32_t time,
                        SectorMask sector_mask) override;
  virtual void allocate_sector(uint32_t time, SectorMask sector_mask);
  virtual void fill(uint32_t time, SectorMask sector_mask) override;
  virtual bool is_valid_line() override;
  virtual bool is_invalid_line() override;
  virtual bool is_reserved_line() override;
  virtual bool is_modified_line() override;
  virtual SectorMask get_dirty_mask() override;
  virtual CacheBlockState get_status(SectorMask mask) override;
  virtual void set_status(CacheBlockState status, SectorMask mask) override;
  virtual bool is_readable(SectorMask mask) override;
  virtual uint64_t get_last_access_time() override;
  virtual uint64_t get_alloc_time() override;
  virtual void set_ignore_on_fill(bool ignore, SectorMask sector_mask) override;
  virtual void set_modified_on_fill(bool modified,
                                    SectorMask sector_mask) override;
  virtual void set_readable(bool readable, SectorMask sector_mask) override;
  virtual void set_last_access_time(uint64_t time,
                                    SectorMask sector_mask) override;
  virtual uint32_t get_modified_size() override;

 private:
  uint32_t m_sector_alloc_time[SECTOR_CHUNCK_SIZE] = {0};
  uint32_t m_sector_fill_time[SECTOR_CHUNCK_SIZE] = {0};
  uint32_t m_sector_last_access_time[SECTOR_CHUNCK_SIZE] = {0};
  uint32_t m_sector_size = 0;
  uint32_t m_line_alloc_time = 0;
  uint32_t m_line_fill_time = 0;
  uint32_t m_line_last_access_time = 0;
  CacheBlockState m_status[SECTOR_CHUNCK_SIZE] = {INVALID};
  bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE] = {false};
  bool m_set_modified_on_fill_status[SECTOR_CHUNCK_SIZE] = {false};
  bool m_readable[SECTOR_CHUNCK_SIZE] = {true};

  void init();

  uint32_t get_sector_index(SectorMask sector_mask) {
    assert(sector_mask.count() == 1);
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (sector_mask.to_ulong() & (1 << i)) return i;
    }
    assert(false);
    return 0;
  }
};

class TagArray {
 public:
  TagArray(CacheConfig &config, int core_id, int type_id);
  ~TagArray();
  CacheRequestStatus probe(uint64_t addr, uint32_t &idx, mem_fetch *mf,
                           bool probe_mode = false) const;
  CacheRequestStatus probe(uint64_t addr, uint32_t &idx, SectorMask mask,
                           mem_fetch *mf = NULL, bool probe_mode = false) const;
  CacheRequestStatus access(uint64_t addr, uint32_t time, uint32_t &idx,
                            mem_fetch *mf);
  CacheRequestStatus access(uint64_t addr, uint32_t time, uint32_t &idx,
                            mem_fetch *mf, bool &wb,
                            EvictedBlockInfo &evicted_block);
  void fill(uint64_t addr, uint32_t time, mem_fetch *mf);
  void fill(uint32_t idx, uint32_t time, mem_fetch *mf);
  void fill(uint64_t addr, uint32_t time, SectorMask mask);
  uint32_t size() const { return m_config.get_num_lines(); }
  CacheBlock *get_block(uint32_t idx) const { return m_lines[idx]; }
  void invalidate();

 protected:
  CacheConfig &m_config;
  CacheBlock **m_lines; /* N banks x M sets x assoc lines in total */
  uint32_t m_core_id;
  uint32_t m_type_id;
  uint32_t m_access;
  uint32_t m_miss;
  uint32_t m_pending_hit;
  uint32_t m_res_fail;
  uint32_t m_sector_miss;
  bool is_used;
  void init(int core_id, int type_id);
};

class MshrTable {
 public:
  MshrTable(uint32_t num_entries, uint32_t max_merged)
      : m_num_entries(num_entries), m_max_merged(max_merged) {}
  bool probe(uint64_t block_addr) const;
  bool full(uint64_t block_addr) const;
  void add(uint64_t block_addr, mem_fetch *mf);
  bool busy() const { return false; }
  void mark_ready(uint64_t block_addr, bool &has_atomic);
  bool access_ready() const { return !m_current_response.empty(); }
  mem_fetch *pop_next_access();
  mem_fetch *top_next_access();
  bool is_read_after_write_pending(uint64_t block_addr);
  void print(FILE *fp) const;

 private:
  const unsigned m_num_entries;
  const unsigned m_max_merged;

  struct MshrEntry {
    std::deque<mem_fetch *> m_list;
    bool m_has_atomic = false;
  };
  std::map<uint64_t, MshrEntry> m_table;
  std::map<uint64_t, MshrEntry> m_line_table;
  bool m_current_response_ready;
  std::deque<uint64_t> m_current_response;
};

class Cache {
 public:
  Cache(std::string name, CacheConfig &config, int core_id, int type_id,
        std::queue<mem_fetch*> *to_mem_queue);
  ~Cache() {
    delete m_tag_array;
    delete m_mshrs;
  }
  virtual CacheRequestStatus access(uint64_t addr, uint32_t time, mem_fetch *mf,
                                    std::deque<CacheEvent> &event) = 0;
  virtual void cycle();
  virtual void fill(mem_fetch *mf, uint32_t time);
  virtual bool waiting_for_fill(mem_fetch *mf);
  virtual bool access_ready() { return m_mshrs->access_ready(); }
  virtual mem_fetch *pop_next_access() { return m_mshrs->pop_next_access(); }
  virtual mem_fetch *top_next_access() { return m_mshrs->top_next_access(); }
  virtual void invalidate() { m_tag_array->invalidate(); }

  virtual bool data_port_free() {
    return m_bandwidth_management.data_port_free();
  }
  virtual bool fill_port_free() {
    return m_bandwidth_management.fill_port_free();
  }
  // virtual bool miss_queue_size(bool from_ndp);
  virtual void force_tag_access(uint64_t addr, uint32_t time, SectorMask mask) {
    m_tag_array->fill(addr, time, mask);
  }
  virtual CacheStats get_stats() const { return m_stats; }
  virtual void print_cache_stats() {}
  
 protected:
  uint32_t m_id;
  std::string m_name;
  CacheConfig &m_config;
  TagArray *m_tag_array;
  MshrTable *m_mshrs;
  std::deque<mem_fetch*> m_miss_queue;
  std::queue<mem_fetch*> *m_to_mem_queue;
  CacheStats m_stats;
  struct ExtraMfFields {
    bool m_valid = false;
    uint64_t m_block_addr;
    uint64_t m_addr;
    uint32_t m_cache_index;
    uint32_t m_data_size;
    uint32_t pending_read;
  };
  class BandwidthManagement {
   public:
    BandwidthManagement(CacheConfig &config) : m_config(config) {}
    void use_data_port(mem_fetch *mf, CacheRequestStatus outcome,
                       const std::deque<CacheEvent> &events);
    void use_fill_port(mem_fetch *mf);
    void replenish_port_bandwidth();
    bool data_port_free() const;
    bool fill_port_free() const;

   protected:
    const CacheConfig &m_config;
    int m_data_port_occupied_cycles = 0;
    int m_fill_port_occupied_cycles = 0;
  };

  std::map<mem_fetch *, ExtraMfFields> m_extra_mf_fields;
  BandwidthManagement m_bandwidth_management;

 protected:
  /// Checks whether this request can be handled on this cycle. num_miss equals
  /// max # of misses to be handled on this cycle
  bool miss_queue_full(uint32_t num_misses) {
    return (m_miss_queue.size() + num_misses) > m_config.get_miss_queue_size();
    ;
  }
  // Read miss handler without write back
  void send_read_request(uint64_t addr, uint64_t block_addr,
                         uint32_t cache_index, mem_fetch *mf, uint32_t time,
                         bool &do_miss, std::deque<CacheEvent> &events,
                         bool read_only, bool wa);
  // Read miss handler. Check MSHR hit or avaiable
  void send_read_request(uint64_t addr, uint64_t block_addr,
                         uint32_t cache_index, mem_fetch *mf, uint32_t time,
                         bool &do_miss, bool &wb, EvictedBlockInfo &eviced,
                         std::deque<CacheEvent> &events, bool read_only,
                         bool wa);
};

class ReadOnlyCache : public Cache {
 public:
  ReadOnlyCache(std::string name, CacheConfig &config, int core_id, int type_id,
                std::queue<mem_fetch*> *to_mem_queue)
      : Cache(name, config, core_id, type_id, to_mem_queue) {}

  virtual CacheRequestStatus access(uint64_t addr, uint32_t time, mem_fetch *mf,
                                    std::deque<CacheEvent> &event) override;
};

class DataCache : public Cache {
 public:
  DataCache(std::string name, CacheConfig &config, int core_id, int type_id,
            std::queue<mem_fetch*> *to_mem_queue, bool is_l1 = false)
      : Cache(name, config, core_id, type_id, to_mem_queue) {
    init();
    m_write_alloc_type = L2_CACHE_WA;
    m_write_back_type = L2_CACHE_WB;
  }
  virtual void init();
  virtual void print_cache_stats();
  virtual CacheRequestStatus access(uint64_t addr, uint32_t time, mem_fetch *mf,
                                    std::deque<CacheEvent> &event) override;
 protected:
  mem_access_type m_write_alloc_type;
  mem_access_type m_write_back_type;
  CacheRequestStatus process_tag_probe(bool wr, CacheRequestStatus status,
                                       uint64_t addr, uint32_t cache_index,
                                       mem_fetch *mf, uint32_t time,
                                       std::deque<CacheEvent> &events);
  // Functions for data cache access
  /// Sends write request to lower level memory (write or writeback)
  void send_write_request(mem_fetch *mf, CacheEvent request, uint32_t time,
                          std::deque<CacheEvent> &events);
  void write_back(EvictedBlockInfo &evicted, uint32_t time, std::deque<CacheEvent> &events);

  CacheRequestStatus (DataCache::*m_wr_hit)(uint64_t addr, uint32_t cache_index,
                                            mem_fetch *mf, uint32_t time,
                                            std::deque<CacheEvent> &event,
                                            CacheRequestStatus status);
  CacheRequestStatus (DataCache::*m_wr_miss)(uint64_t addr,
                                             uint32_t cache_index,
                                             mem_fetch *mf, uint32_t time,
                                             std::deque<CacheEvent> &event,
                                             CacheRequestStatus status);
  CacheRequestStatus (DataCache::*m_rd_hit)(uint64_t addr, uint32_t cache_index,
                                            mem_fetch *mf, uint32_t time,
                                            std::deque<CacheEvent> &event,
                                            CacheRequestStatus status);
  CacheRequestStatus (DataCache::*m_rd_miss)(uint64_t addr,
                                             uint32_t cache_index,
                                             mem_fetch *mf, uint32_t time,
                                             std::deque<CacheEvent> &event,
                                             CacheRequestStatus status);

  // Function pointers for different cache access
  // Write hit
  CacheRequestStatus wr_hit_wb(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write hit with write back
  CacheRequestStatus wr_hit_wt(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write hit with write through
  CacheRequestStatus wr_hit_we(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write hit with write evict
  CacheRequestStatus wr_hit_global_we_local_wb(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write hit with write evict for global and
                                   // write back for local
  // Write miss
  CacheRequestStatus wr_miss_wa_naive(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write allocate send write and read requsts
  CacheRequestStatus wr_miss_wa_lazy_fetch_on_read(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // write allocate with read-fetch-only
  CacheRequestStatus wr_miss_wa_write_validate(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus
          status);  // write-allocate that writes with no read fetch
  CacheRequestStatus wr_miss_no_wa(
      uint64_t addr, uint32_t cache_index, mem_fetch *mf, uint32_t time,
      std::deque<CacheEvent> &event,
      CacheRequestStatus status);  // no write allocate

  // Read hit
  CacheRequestStatus rd_hit_base(uint64_t addr, uint32_t cache_index,
                                 mem_fetch *mf, uint32_t time,
                                 std::deque<CacheEvent> &event,
                                 CacheRequestStatus status);  // read hit base

  // Read miss
  CacheRequestStatus rd_miss_base(uint64_t addr, uint32_t cache_index,
                                  mem_fetch *mf, uint32_t time,
                                  std::deque<CacheEvent> &event,
                                  CacheRequestStatus status);  // read miss base
};

#endif