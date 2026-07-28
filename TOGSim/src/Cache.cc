#include "Cache.h"
#include "Hashing.h"

unsigned int LOGB2(unsigned int v) {
  unsigned int shift;
  unsigned int r;
  r = 0;
  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;
  return r;
}

void CacheConfig::init(std::string config) {
  assert(config.size() > 0);
  char cache_type, evict_policy, write_policy, alloc_policy, write_alloc_policy,
      sif;
  char mshr_type;
  // sif : sector index function
  int ntok =
      sscanf(config.c_str(), "%c:%u:%u:%u,%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u",
             &cache_type, &m_nset, &m_line_size, &m_assoc, &m_sector_size, &evict_policy,
             &write_policy, &alloc_policy, &write_alloc_policy, &sif,
             &mshr_type, &m_mshr_entries, &m_mshr_max_merge, &m_miss_queue_size,
             &m_result_fifo_entries, &m_data_port_width);
  assert(ntok >= 12);
  m_valid = true;
  m_cache_type = CacheTypeMap[cache_type];
  m_evict_policy = EvictPolicyMap[evict_policy];
  m_write_policy = WritePolicyMap[write_policy];
  m_alloc_policy = AllocationPolicyMap[alloc_policy];
  m_write_alloc_policy = WriteAllocatePolicyMap[write_alloc_policy];
  m_set_index_function = SetIndexFunctionMap[sif];
  m_mshr_type = MshrConfigMap[mshr_type];
  m_line_size_log2 = LOGB2(m_line_size);
  m_nset_log2 = LOGB2(m_nset);
  m_atom_size = m_cache_type == SECTOR ? m_sector_size : m_line_size;
  m_sector_size_log2 = LOGB2(m_sector_size);
  m_origin_assoc = m_assoc;
  m_origin_nset = m_nset;
}

uint32_t CacheConfig::get_set_index(uint64_t addr) const {
  return hash_function(addr);
}

uint64_t CacheConfig::get_tag(uint64_t addr) const {
  return addr & ~(uint64_t)(m_line_size - 1);
}

uint64_t CacheConfig::get_block_addr(uint64_t addr) const {
  return addr & ~(uint64_t)(m_line_size - 1);
}

uint64_t CacheConfig::get_mshr_addr(uint64_t addr) const {
  return addr & ~(uint64_t)(m_atom_size - 1);
}

uint32_t CacheConfig::hash_function(uint64_t addr) const {
  uint32_t set_index = 0;
  switch (m_set_index_function) {
    case LINEAR_SET_FUNCTION:
      set_index = (addr >> m_line_size_log2) & (m_nset - 1);
      break;
    case BITWISE_XORING_FUNCTION: {
      uint64_t higher_bits = addr > (m_line_size_log2 + m_nset_log2);
      uint32_t index = (addr >> m_line_size_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
    } break;
    case HASH_IPOLY_FUNCTION: {
      uint64_t higher_bits = addr > (m_line_size_log2 + m_nset_log2);
      uint32_t index = (addr >> m_line_size_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
    } break;
    case CUSTOM_SET_FUNCTION:
      break;
    default:
      assert(0);
  }
  return set_index;
}

/* Normal Cache Block */
void LineCacheBlock::allocate(uint64_t tag, uint64_t block_addr, uint32_t time,
                              SectorMask mask) {
  m_tag = tag;
  m_block_addr = block_addr;
  m_alloc_time = time;
  m_last_access_time = time;
  m_fill_time = 0;
  m_status = RESERVED;
  m_ignore_on_fill_status = false;
  m_set_modified_on_fill = false;
}

void LineCacheBlock::fill(uint32_t time, SectorMask) {
  m_fill_time = time;
  m_status = m_set_modified_on_fill ? MODIFIED : VALID;
}

SectorMask LineCacheBlock::get_dirty_mask() {
  SectorMask dirty_mask;
  dirty_mask.reset();
  if (m_status == MODIFIED)
    dirty_mask.set();
  return dirty_mask;
}

/* Sector Cache Block */
void SectorCacheBlock::allocate(uint64_t tag, uint64_t block_addr,
                                uint32_t time, SectorMask sector_mask) {
  // Allocate line
  init();
  m_tag = tag;
  m_block_addr = block_addr;
  uint32_t sidx = get_sector_index(sector_mask);
  m_sector_alloc_time[sidx] = time;
  m_sector_last_access_time[sidx] = time;
  m_line_alloc_time = time;
  m_line_last_access_time = time;
}

void SectorCacheBlock::allocate_sector(uint32_t time, SectorMask sector_mask) {
  assert(is_valid_line());
  uint32_t sidx = get_sector_index(sector_mask);
  m_sector_alloc_time[sidx] = time;
  m_sector_last_access_time[sidx] = time;
  m_line_last_access_time = time;
  m_set_modified_on_fill_status[sidx] = m_status[sidx] == MODIFIED ? true : false; 
  m_status[sidx] = RESERVED;
  m_ignore_on_fill_status[sidx] = false;
  m_readable[sidx] = true;
}

void SectorCacheBlock::fill(uint32_t time, SectorMask sector_mask) {
  uint32_t sidx = get_sector_index(sector_mask);
  m_status[sidx] = m_set_modified_on_fill_status[sidx] ? MODIFIED : VALID;
  m_sector_fill_time[sidx] = time;
  m_line_fill_time = time;
}

bool SectorCacheBlock::is_valid_line() { return !(is_invalid_line()); }

bool SectorCacheBlock::is_invalid_line() {
  // all the sectors should be invalid
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
    if (m_status[i] != INVALID) return false;
  }
  return true;
}

bool SectorCacheBlock::is_reserved_line() {
  // all the sectors should be invalid
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
    if (m_status[i] == RESERVED) return true;
  }
  return false;
}

bool SectorCacheBlock::is_modified_line() {
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
    if (m_status[i] == MODIFIED) return true;
  }
  return false;
}

SectorMask SectorCacheBlock::get_dirty_mask() {
  SectorMask dirty_mask;
  dirty_mask.reset();
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
    if (m_status[i] == MODIFIED) dirty_mask.set(i);
  }
  return dirty_mask;
}

void SectorCacheBlock::init() {
  for (int i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    m_sector_alloc_time[i] = 0;
    m_sector_fill_time[i] = 0;
    m_sector_last_access_time[i] = 0;
    m_status[i] = INVALID;
    m_ignore_on_fill_status[i] = false;
    m_set_modified_on_fill_status[i] = false;
    m_readable[i] = true;
  }
  m_line_alloc_time = 0;
  m_line_fill_time = 0;
  m_line_last_access_time = 0;
}

CacheBlockState SectorCacheBlock::get_status(SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  return m_status[sidx];
}

void SectorCacheBlock::set_status(CacheBlockState status, SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  m_status[sidx] = status;
}

bool SectorCacheBlock::is_readable(SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  return m_readable[sidx];
}

uint64_t SectorCacheBlock::get_last_access_time() {
  return m_line_last_access_time;
}

uint64_t SectorCacheBlock::get_alloc_time() { return m_line_alloc_time; }

void SectorCacheBlock::set_ignore_on_fill(bool ignore, SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  m_ignore_on_fill_status[sidx] = ignore;
}

void SectorCacheBlock::set_modified_on_fill(bool modified, SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  m_set_modified_on_fill_status[sidx] = modified;
}

void SectorCacheBlock::set_readable(bool readable, SectorMask mask) {
  uint32_t sidx = get_sector_index(mask);
  m_readable[sidx] = readable;
}

void SectorCacheBlock::set_last_access_time(uint64_t time,
                                            SectorMask sector_mask) {
  m_line_last_access_time = time;
  uint32_t sidx = get_sector_index(sector_mask);
  m_sector_last_access_time[sidx] = time;
}

uint32_t SectorCacheBlock::get_modified_size() {
  uint32_t modified_size = 0;
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
    if (m_status[i] == MODIFIED) modified_size++;
  }
  return modified_size * m_sector_size;
}

/*Tag Array*/
TagArray::TagArray(CacheConfig &config, int core_id, int type_id)
    : m_config(config) {
  uint32_t cache_lines_num = config.get_num_lines();
  m_lines = new CacheBlock *[cache_lines_num];
  for (uint32_t i = 0; i < cache_lines_num; ++i) {
    if (config.get_cache_type() == SECTOR)
      m_lines[i] = new SectorCacheBlock(config.get_sector_size());
    else if (config.get_cache_type() == NORMAL)
      m_lines[i] = new LineCacheBlock(config.get_sector_size());
    else
      assert(0);
  }
  init(core_id, type_id);
}

TagArray::~TagArray() {
  uint32_t cache_lines_num = m_config.get_num_lines();
  for (uint32_t i = 0; i < cache_lines_num; ++i) {
    delete m_lines[i];
  }
  delete[] m_lines;
}

CacheRequestStatus TagArray::probe(uint64_t addr, uint32_t &idx, mem_fetch *mf,
                                   bool probe_mode) const {
  SectorMask sector_mask = mf->get_access_sector_mask();
  return probe(addr, idx, sector_mask, mf, probe_mode);
}

CacheRequestStatus TagArray::probe(uint64_t addr, uint32_t &idx,
                                   SectorMask mask, mem_fetch *mf,
                                   bool probe_mode) const {
  int set_index = m_config.get_set_index(addr);
  uint64_t tag = m_config.get_tag(addr);
  uint32_t valid_line = (uint32_t)-1;
  uint32_t invalid_line = (uint32_t)-1;
  uint64_t valid_timestamp = (uint64_t)-1;
  bool all_reserved = true;
  for (uint32_t way = 0; way < m_config.get_num_assoc(); way++) {
    uint32_t index = set_index * m_config.get_num_assoc() + way;
    CacheBlock *line = m_lines[index];

    // Handle tag matched case
    if (line->match_tag(tag)) {
      idx = index;
      if (line->get_status(mask) == RESERVED) {
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID ||
                 (line->get_status(mask) == MODIFIED &&
                  line->is_readable(mask))) {
        return HIT;
      } else if ((line->get_status(mask) == MODIFIED &&
                  !line->is_readable(mask)) ||
                 (line->is_valid_line() && line->get_status(mask) == INVALID)) {
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    } else if (!line->is_reserved_line()) {
      all_reserved = false;
      if (line->is_invalid_line()) {
        invalid_line = index;
        continue;
      }

      // Choose cacheline for eviction
      if (m_config.get_evict_policy() == LRU) {
        if (line->get_last_access_time() < valid_timestamp) {
          valid_timestamp = line->get_last_access_time();
          valid_line = index;
        }
      } else if (m_config.get_evict_policy() == FIFO) {
        if (line->get_alloc_time() < valid_timestamp) {
          valid_timestamp = line->get_alloc_time();
          valid_line = index;
        }
      }
    }
  }

  // All target cachelines are reserved
  if (all_reserved) {
    assert(m_config.get_alloc_policy() == ON_MISS);
    return RESERVATION_FAIL;
  }

  if (invalid_line != (uint32_t)-1) {
    idx = invalid_line;
  } else if (valid_line != (uint32_t)-1) {
    idx = valid_line;
  } else {
    assert(0);
  }
  return MISS;
}

CacheRequestStatus TagArray::access(uint64_t addr, uint32_t time, uint32_t &idx,
                                    mem_fetch *mf) {
  bool wb = false;
  EvictedBlockInfo evicted;
  return access(addr, time, idx, mf, wb, evicted);
}

CacheRequestStatus TagArray::access(uint64_t addr, uint32_t time, uint32_t &idx,
                                    mem_fetch *mf, bool &wb,
                                    EvictedBlockInfo &evicted) {
  is_used = true;
  m_access++;
  SectorMask sector_mask = mf->get_access_sector_mask();
  uint64_t tag = m_config.get_tag(addr);
  uint64_t block_addr = m_config.get_block_addr(addr);
  CacheRequestStatus status = probe(addr, idx, mf);
  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
      break;
    case HIT:
      m_lines[idx]->set_last_access_time(time, sector_mask);
      break;
    case SECTOR_MISS:
      assert(m_config.get_cache_type() == SECTOR);
      m_sector_miss++;
      if (m_config.get_alloc_policy() == ON_MISS) {
        ((SectorCacheBlock *)m_lines[idx])->allocate_sector(time, sector_mask);
      }
      break;
    case MISS:
      m_miss++;
      if (m_config.get_alloc_policy() == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          evicted.set_info(m_lines[idx]->get_block_addr(), m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_status(sector_mask));
        }
        m_lines[idx]->allocate(tag, block_addr, time, sector_mask);
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      break;
  }
  return status;
}

void TagArray::fill(uint64_t addr, uint32_t time, mem_fetch *mf) {
  fill(addr, time, mf->get_access_sector_mask());
}

void TagArray::fill(uint32_t index, uint32_t time, mem_fetch *mf) {
  assert(m_config.get_alloc_policy() == ON_MISS);
  m_lines[index]->fill(time, mf->get_access_sector_mask());
}

void TagArray::fill(uint64_t addr, uint32_t time, SectorMask mask) {
  uint32_t idx;
  CacheRequestStatus status = probe(addr, idx, mask);
  if (status == MISS) {
    m_lines[idx]->allocate(m_config.get_tag(addr),
                           m_config.get_block_addr(addr), time, mask);
  } else if (status == SECTOR_MISS) {
    assert(m_config.get_cache_type() == SECTOR);
    ((SectorCacheBlock *)m_lines[idx])->allocate_sector(time, mask);
  }
  m_lines[idx]->fill(time, mask);
}

void TagArray::invalidate() {
  if (!is_used) return;
  for (uint32_t i = 0; i < m_config.get_num_lines(); i++) {
    for (uint32_t j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
      m_lines[i]->set_status(INVALID, SectorMask().set(j));
    }
  }
}

void TagArray::init(int core_id, int type_id) {
  m_core_id = core_id;
  m_type_id = type_id;
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  is_used = false;
}

/* MSHR Table */
bool MshrTable::probe(uint64_t block_addr) const {
  return m_table.find(block_addr) != m_table.end();
}

bool MshrTable::full(uint64_t block_addr) const {
  if (probe(block_addr))
    return m_table.at(block_addr).m_list.size() >= m_max_merged;
  else
    return m_table.size() >= m_num_entries;
}

void MshrTable::add(uint64_t block_addr, mem_fetch *mf) {
  assert(!full(block_addr));
  m_table[block_addr].m_list.push_back(mf);
  if (mf->is_atomic()) {
    m_table[block_addr].m_has_atomic = true;
  }
}

void MshrTable::mark_ready(uint64_t block_addr, bool &has_atomic) {
  assert(probe(block_addr));
  has_atomic = m_table[block_addr].m_has_atomic;
  m_current_response.push_back(block_addr);
  }

mem_fetch *MshrTable::pop_next_access() {
  assert(access_ready());
  uint64_t block_addr = m_current_response.front();
  assert(probe(block_addr));
  mem_fetch *mf = m_table[block_addr].m_list.front();
  m_table[block_addr].m_list.pop_front();
  if (m_table[block_addr].m_list.empty()) {
    m_table.erase(block_addr);
    m_current_response.pop_front();
  }
  return mf;
}

mem_fetch *MshrTable::top_next_access() {
  assert(access_ready());
  uint64_t block_addr = m_current_response.front();
  assert(probe(block_addr));
  mem_fetch *mf = m_table[block_addr].m_list.front();
  return mf;
}

bool MshrTable::is_read_after_write_pending(uint64_t block_addr) {
  std::deque<mem_fetch *> list = m_table[block_addr].m_list;
  bool write_found = false;
  for (auto it = list.begin(); it != list.end(); ++it) {
    if ((*it)->is_write()) {
      write_found = true;  // Pending write
    } else if (write_found) {
      return true;  // Pending read after write
    }
  }
  return false;
}

void MshrTable::print(FILE *fp) const {

}

/* Cache */
Cache::Cache(std::string name, CacheConfig &config, int core_id, int type_id,
             std::queue<mem_fetch*> *to_mem_queue)
    : m_config(config), m_bandwidth_management(config) {
  m_tag_array = new TagArray(config, core_id, type_id);
  m_mshrs = new MshrTable(config.get_mshr_entries(),
                                        config.get_mshr_max_merge());
  m_name = name + std::to_string(core_id);
  m_id = core_id;
  m_to_mem_queue = to_mem_queue;
}

void Cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    m_to_mem_queue->push(mf);
    m_miss_queue.pop_front();
  }
  m_bandwidth_management.replenish_port_bandwidth();
}

void Cache::fill(mem_fetch *mf, uint32_t time) {
  if (m_config.get_mshr_config() == SECTOR_ASSOC) {
    assert(mf->get_original_mf());
    assert(m_extra_mf_fields.find(mf->get_original_mf()) !=
           m_extra_mf_fields.end());
    m_extra_mf_fields[mf->get_original_mf()].pending_read--;
    if (m_extra_mf_fields[mf->get_original_mf()].pending_read > 0) {
      delete mf;
      return;
    } else {
      mem_fetch *tmp = mf;
      mf = mf->get_original_mf();
      delete tmp;
    }
  }
  assert(m_extra_mf_fields.find(mf) != m_extra_mf_fields.end());
  ExtraMfFields field = m_extra_mf_fields[mf];
  mf->set_data_size(field.m_data_size);
  mf->set_addr(field.m_addr);
  if (m_config.get_alloc_policy() == ON_MISS) {
    m_tag_array->fill(field.m_cache_index, time, mf);
  } else if (m_config.get_alloc_policy() == ON_FILL) {
    m_tag_array->fill(field.m_block_addr, time, mf);
  }
  bool has_atomic = false;
  m_mshrs->mark_ready(field.m_block_addr, has_atomic);
  if (has_atomic) {
    assert(m_config.get_alloc_policy() == ON_MISS);
    CacheBlock *block = m_tag_array->get_block(field.m_cache_index);
    if(!block->is_modified_line()) {
      // m_tag_array->inc_dirty(); // TODO
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
  }
  m_extra_mf_fields.erase(mf);
  m_bandwidth_management.use_fill_port(mf);
}

bool Cache::waiting_for_fill(mem_fetch *mf) {
  return m_extra_mf_fields.find(mf) != m_extra_mf_fields.end();
}

void Cache::send_read_request(uint64_t addr, uint64_t block_addr,
                              uint32_t cache_index, mem_fetch *mf,
                              uint32_t time, bool &do_miss,
                              std::deque<CacheEvent> &events, bool read_only,
                              bool ws) {
  bool wb = false;
  EvictedBlockInfo evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, read_only, ws);
}

void Cache::send_read_request(uint64_t addr, uint64_t block_addr,
                              uint32_t cache_index, mem_fetch *mf,
                              uint32_t time, bool &do_miss, bool &wb,
                              EvictedBlockInfo &evicted,
                              std::deque<CacheEvent> &events, bool read_only,
                              bool wa) {
  new_addr_type mshr_addr = m_config.get_mshr_addr(addr);
  bool mshr_hit = m_mshrs->probe(mshr_addr);
  bool mshr_avail = !m_mshrs->full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, mf, wb, evicted);
    m_mshrs->add(mshr_addr, mf);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
    do_miss = true;
  } else if (!mshr_hit && mshr_avail && !miss_queue_full(0)) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, mf, wb, evicted);
    m_mshrs->add(mshr_addr, mf);
    m_extra_mf_fields[mf] = ExtraMfFields();
    m_extra_mf_fields[mf].m_valid = true;
    m_extra_mf_fields[mf].m_block_addr = mshr_addr;
    m_extra_mf_fields[mf].m_addr = mf->get_addr();
    m_extra_mf_fields[mf].m_cache_index = cache_index;
    m_extra_mf_fields[mf].m_data_size = mf->get_data_size();
    m_extra_mf_fields[mf].pending_read = m_config.get_mshr_config() == SECTOR_ASSOC
                            ? m_config.get_line_size() / m_config.get_sector_size()
                            : 0;
    mf->set_data_size(m_config.get_atom_size());
    // assert(m_config.get_atom_size() <= PACKET_SIZE); //TODO: for now, it should be true
    mf->set_addr(mshr_addr);
    m_miss_queue.push_back(mf);
    if (!wa) events.push_back(CacheEvent(READ_REQUEST_SENT));
    do_miss = true;
  } else if (mshr_hit && !mshr_avail) {
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENTRY_FAIL);
  } else if (!mshr_hit && !mshr_avail) {
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENTRY_FAIL);
  }
}

void Cache::BandwidthManagement::use_data_port(
    mem_fetch *mf, CacheRequestStatus outcome,
    const std::deque<CacheEvent> &events) {
  uint32_t data_size = mf->get_data_size();
  uint32_t port_width = m_config.get_data_port_width();
  uint32_t data_cycles = 0;
  CacheEvent event;
  switch (outcome) {
    case HIT:
      data_cycles = data_size / port_width + ((data_size % port_width) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
      break;
    case HIT_RESERVED:
    case MISS:
      if (CacheEvent::was_writeback_sent(events, event)) {
        data_cycles = event.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
      break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      break;
    default:
      assert(0);
  }
}

void Cache::BandwidthManagement::use_fill_port(mem_fetch *mf) {
  unsigned fill_cycles =
      m_config.get_atom_size() / m_config.get_data_port_width();
  m_fill_port_occupied_cycles += fill_cycles;
}

void Cache::BandwidthManagement::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles--;
  }
  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles--;
  }
}

bool Cache::BandwidthManagement::data_port_free() const {
  return true; // ignore this feature
}

bool Cache::BandwidthManagement::fill_port_free() const {
  return true;
}

/* Read-only Cache */
CacheRequestStatus ReadOnlyCache::access(uint64_t addr, uint32_t time,
                                         mem_fetch *mf,
                                         std::deque<CacheEvent> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_size());
  assert(m_config.get_write_policy() == READ_ONLY);
  assert(!mf->is_write());
  uint64_t block_addr = m_config.get_block_addr(addr);
  uint32_t cache_index = (uint32_t)-1;
  CacheRequestStatus status =
      m_tag_array->probe(block_addr, cache_index, mf, true);
  CacheRequestStatus cache_status = RESERVATION_FAIL;
  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index, mf);
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));

  m_bandwidth_management.use_data_port(mf, cache_status, events);
  return cache_status;
}

/* Data Cache */
void DataCache::init() {
  m_rd_hit = &DataCache::rd_hit_base;
  m_rd_miss = &DataCache::rd_miss_base;
  switch (m_config.get_write_policy()) {
    case READ_ONLY:
      assert(0);  // Data cache cannot be read only
    case WRITE_BACK:
      m_wr_hit = &DataCache::wr_hit_wb;
      break;
    case WRITE_THROUGH:
      m_wr_hit = &DataCache::wr_hit_wt;
      break;
    case WRITE_EVICT:
      m_wr_hit = &DataCache::wr_hit_we;
      break;
    default:
      assert(0);
  }
  switch (m_config.get_write_alloc_policy()) {
    case NO_WRITE_ALLOCATE:
      m_wr_miss = &DataCache::wr_miss_no_wa;
      break;
    case WRITE_ALLOCATE:
      m_wr_miss = &DataCache::wr_miss_wa_naive;
      break;
    default:
      assert(0);
  }
}

void DataCache::print_cache_stats() {
  uint64_t hit = m_stats.get_interval_hit();
  uint64_t miss = m_stats.get_interval_miss();
  if (m_id == 0) {
    spdlog::info("NDP {:2}: average Data Cache Hit : {}, Miss : {} , Hit Raito : {:.2f}\%", m_id,
                 hit, miss, ((float)hit) / (hit + miss) * 100);
  } else {
    spdlog::debug("NDP {:2}: average Data Cache Hit : {}, Miss : {} , Hit Raito : {:.2f}\%", m_id,
                 hit, miss, ((float)hit) / (hit + miss) * 100);
  }
}

CacheRequestStatus DataCache::access(uint64_t addr, uint32_t time,
                                     mem_fetch *mf,
                                     std::deque<CacheEvent> &events) {
  bool wr = mf->is_write();
  uint64_t block_addr = m_config.get_block_addr(addr);
  uint32_t cache_index = (uint32_t)-1;
  CacheRequestStatus probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, true);
  CacheRequestStatus access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status));
  return access_status;
}

CacheRequestStatus DataCache::process_tag_probe(bool wr,
                                                CacheRequestStatus probe_status,
                                                uint64_t addr,
                                                uint32_t cache_index,
                                                mem_fetch *mf, uint32_t time,
                                                std::deque<CacheEvent> &events) {
  CacheRequestStatus access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL ||
               (probe_status == RESERVATION_FAIL &&
                m_config.get_write_alloc_policy() == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  }
  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

void DataCache::send_write_request(mem_fetch *mf, CacheEvent request,
                                   uint32_t time,
                                   std::deque<CacheEvent> &events) {
  events.push_back(request);
  m_miss_queue.push_back(mf);
}

void DataCache::write_back(EvictedBlockInfo &evicted, uint32_t time, std::deque<CacheEvent> &events) {
  auto packet_size = m_config.get_atom_size();
  for(int i = 0; i < evicted.m_modified_size / packet_size; i++) {
    uint64_t evicted_addr = evicted.m_block_addr + i * packet_size;
    mem_fetch *wb_mf =
        new mem_fetch(evicted_addr, m_write_back_type, WRITE_REQUEST,
                      packet_size);
    wb_mf->set_dirty_mask(evicted.m_dirty_mask);
    send_write_request(wb_mf, CacheEvent(WRITE_BACK_REQUEST_SENT, evicted),
                       time, events);
  }
}

/*** WRITE-hit functions (Set by config file) ***/
// Write hit: Write back
CacheRequestStatus DataCache::wr_hit_wb(uint64_t addr, uint32_t cache_index,
                                        mem_fetch *mf, uint32_t time,
                                        std::deque<CacheEvent> &events,
                                        CacheRequestStatus status) {
  uint64_t block_addr = m_config.get_block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  CacheBlock *block = m_tag_array->get_block(cache_index);
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  return HIT;
}

// Write hit: Write through
CacheRequestStatus DataCache::wr_hit_wt(uint64_t addr, uint32_t cache_index,
                                        mem_fetch *mf, uint32_t time,
                                        std::deque<CacheEvent> &events,
                                        CacheRequestStatus status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }
  uint64_t block_addr = m_config.get_block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  CacheBlock *block = m_tag_array->get_block(cache_index);
  block->set_status(MODIFIED, mf->get_access_sector_mask());

  // Generate a write-through
  send_write_request(mf, CacheEvent(WRITE_REQUEST_SENT), time, events);
  return HIT;
}

// Write hit: Write evict
CacheRequestStatus DataCache::wr_hit_we(uint64_t addr, uint32_t cache_index,
                                        mem_fetch *mf, uint32_t time,
                                        std::deque<CacheEvent> &events,
                                        CacheRequestStatus status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }
  CacheBlock *block = m_tag_array->get_block(cache_index);
  send_write_request(mf, CacheEvent(WRITE_REQUEST_SENT), time, events);
  block->set_status(INVALID, mf->get_access_sector_mask());
  return HIT;
}

/*** WRITE-miss functions (Set by config file) ***/
// Write miss: Write allocate naive
CacheRequestStatus DataCache::wr_miss_wa_naive(uint64_t addr,
                                               uint32_t cache_index,
                                               mem_fetch *mf, uint32_t time,
                                               std::deque<CacheEvent> &events,
                                               CacheRequestStatus status) {
  uint64_t block_addr = m_config.get_block_addr(addr);
  uint64_t mshr_addr = m_config.get_mshr_addr(addr);
  bool mshr_hit = m_mshrs->probe(mshr_addr);
  bool mshr_avail = !m_mshrs->full(mshr_addr);
  if (miss_queue_full(2)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  } else if (mshr_hit && !mshr_avail) {
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENTRY_FAIL);
    return RESERVATION_FAIL;
  } else if (!mshr_hit && !mshr_avail) {
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENTRY_FAIL);
    return RESERVATION_FAIL;
  }
  send_write_request(mf, CacheEvent(WRITE_REQUEST_SENT), time, events);
  mem_fetch *new_mf = new mem_fetch(
      mf->get_addr(), m_write_alloc_type, READ_REQUEST, m_config.get_atom_size());
  new_mf->set_access_sector_mask(mf->get_access_sector_mask());
  new_mf->set_core_id(mf->get_core_id());
  bool do_miss = false;
  bool wb = false;
  EvictedBlockInfo evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, new_mf, time, do_miss, wb,
                    evicted, events, false, true);
  if (do_miss) {
    if (wb && (m_config.get_write_policy() != WRITE_THROUGH)) {
      assert(status == MISS);
      write_back(evicted, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

// Write miss: Write allocate no write allocate
CacheRequestStatus DataCache::wr_miss_no_wa(uint64_t addr, uint32_t cache_index,
                                            mem_fetch *mf, uint32_t time,
                                            std::deque<CacheEvent> &events,
                                            CacheRequestStatus status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }
  send_write_request(mf, CacheEvent(WRITE_REQUEST_SENT), time, events);
  return MISS;
}

CacheRequestStatus DataCache::rd_hit_base(uint64_t addr, uint32_t cache_index,
                                          mem_fetch *mf, uint32_t time,
                                          std::deque<CacheEvent> &events,
                                          CacheRequestStatus status) {
  uint64_t block_addr = m_config.get_block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  if (mf->is_atomic()) {
    CacheBlock *block = m_tag_array->get_block(cache_index);
    block->set_status(MODIFIED, mf->get_access_sector_mask());
  }
  return HIT;
}

CacheRequestStatus DataCache::rd_miss_base(uint64_t addr, uint32_t cache_index,
                                           mem_fetch *mf, uint32_t time,
                                           std::deque<CacheEvent> &events,
                                           CacheRequestStatus status) {
  if (miss_queue_full(1)) {
    mf->current_state = "MISS_QUEUE_FULL";
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }
  uint64_t block_addr = m_config.get_block_addr(addr);
  bool do_miss = false;
  bool wb = false;
  EvictedBlockInfo evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, true);
  if (do_miss) {
    if (wb && (m_config.get_write_policy() != WRITE_THROUGH)) {
      write_back(evicted, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}