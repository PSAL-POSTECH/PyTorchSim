#include "Cache_stats.h"
#include "Memfetch.h"

CacheStats::CacheStats() {
  m_stats.resize(NUM_MEM_ACCESS_TYPE);
  m_fail_stats.resize(NUM_MEM_ACCESS_TYPE);
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_fail_stats[i].resize(NUM_CACHE_RESERVATION_FAIL_REASON, 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;

  m_prev_hit = 0;
  m_prev_miss = 0;
}

void CacheStats::clear() {
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    std::fill(m_fail_stats[i].begin(), m_fail_stats[i].end(), 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void CacheStats::inc_stats(int access_type, int access_outcome) {
  assert(check_valid(access_type, access_outcome));
  m_stats[access_type][access_outcome]++;
}

void CacheStats::inc_fail_stats(int access_type, int fail_outcome) {
  assert(check_fail_valid(access_type, fail_outcome));
  m_fail_stats[access_type][fail_outcome]++;
}

CacheRequestStatus CacheStats::select_stats_status(
    CacheRequestStatus probe, CacheRequestStatus access) const {
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

uint64_t &CacheStats::operator()(int access_type, int access_outcome,
                                 bool fail_outcome) {
  if (fail_outcome) {
    assert(check_fail_valid(access_type, access_outcome));
    return m_fail_stats[access_type][access_outcome];
  } else {
    assert(check_valid(access_type, access_outcome));
    return m_stats[access_type][access_outcome];
  }
}

uint64_t CacheStats::operator()(int access_type, int access_outcome,
                                bool fail_outcome) const {
  if (fail_outcome) {
    assert(check_fail_valid(access_type, access_outcome));
    return m_fail_stats[access_type][access_outcome];
  } else {
    assert(check_valid(access_type, access_outcome));
    return m_stats[access_type][access_outcome];
  }
}

CacheStats CacheStats::operator+(const CacheStats &other) {
  CacheStats sum;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      sum.m_stats[i][j] = m_stats[i][j] + other.m_stats[i][j];
      sum.m_fail_stats[i][j] = m_fail_stats[i][j] + other.m_fail_stats[i][j];
    }
  }
  sum.m_cache_port_available_cycles =
      m_cache_port_available_cycles + other.m_cache_port_available_cycles;
  sum.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + other.m_cache_data_port_busy_cycles;
  sum.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + other.m_cache_fill_port_busy_cycles;
  return sum;
}

CacheStats &CacheStats::operator+=(const CacheStats &other) {
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      m_stats[i][j] += other.m_stats[i][j];
      m_fail_stats[i][j] += other.m_fail_stats[i][j];
    }
  }
  m_cache_port_available_cycles += other.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += other.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += other.m_cache_fill_port_busy_cycles;
  return *this;
}

uint64_t CacheStats::get_hit() const {
  uint64_t hit = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if (j == HIT) hit += m_stats[i][j];
    }
  }
  return hit;
}

uint64_t CacheStats::get_read_hit() const {
  uint64_t hit = 0;
  mem_access_type types[] = {GLOBAL_ACC_R};
  CacheRequestStatus status[] = {HIT, HIT_RESERVED};
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      hit += m_stats[types[i]][status[j]];
    }
  }
  return hit;
}

uint64_t CacheStats::get_write_hit() const {
  uint64_t hit = 0;
  mem_access_type types[] = {GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB};
  CacheRequestStatus status[] = {HIT, HIT_RESERVED};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      hit += m_stats[types[i]][status[j]];
    }
  }
  return hit;
}


uint64_t CacheStats::get_miss() const {
  uint64_t miss = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if (j == MISS || j == SECTOR_MISS) miss += m_stats[i][j];
    }
  }
  return miss;
}

uint64_t CacheStats::get_read_miss() const {
  uint64_t miss = 0;
  mem_access_type types[] = {GLOBAL_ACC_R};
  CacheRequestStatus status[] = {MISS, SECTOR_MISS};
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      miss += m_stats[types[i]][status[j]];
    }
  }
  return miss;
}

uint64_t CacheStats::get_write_miss() const {
  uint64_t miss = 0;
  mem_access_type types[] = {GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB};
  CacheRequestStatus status[] = {MISS, SECTOR_MISS};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      miss += m_stats[types[i]][status[j]];
    }
  }
  return miss;
}

uint64_t CacheStats::get_accesses() const {
  uint64_t access = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if(j == HIT || j == MISS || j == SECTOR_MISS || j == HIT_RESERVED)
        access += m_stats[i][j];
    }
  }
  return access;
}

uint64_t CacheStats::get_interval_hit() {
  uint64_t prev_hit = m_prev_hit;
  m_prev_hit = get_hit();

  return m_prev_hit - prev_hit;
}

uint64_t CacheStats::get_interval_miss() {
  uint64_t prev_miss = m_prev_miss;
  m_prev_miss = get_miss();

  return m_prev_miss - prev_miss;
}

void CacheStats::print_stats(FILE *out, const char *cache_name) const {
  uint64_t hit = get_hit();
  uint64_t miss = get_miss();
  fprintf(out, "\tCache Hit : %lu, Cache Miss : %lu, Hit Ratio : %.2f\n", hit,
          miss, (float)hit / (get_accesses()));
  std::vector<uint32_t> total_access;
  total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    for (int status = 0; status < NUM_CACHE_REQUEST_STATUS; status++) {
      fprintf(out, "\t%s[%s][%s] = %lu\n", cache_name,
              mem_access_type_str[type], cache_request_status_str[status],
              m_stats[type][status]);
      if (status != RESERVATION_FAIL && status != MSHR_HIT)
        total_access[type] += m_stats[type][status];
    }
  }
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    fprintf(out, "\t%s[%s][TOTAL] = %u\n", cache_name,
            mem_access_type_str[type], total_access[type]);
  }
}

void CacheStats::print_fail_stats(FILE *out, const char *cache_name) const {
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    for (int status = 0; status < NUM_CACHE_RESERVATION_FAIL_REASON; status++) {
      fprintf(out, "\t%s[%s][%s] = %lu\n", cache_name,
              mem_access_type_str[type],
              cache_reservation_fail_reason_str[status],
              m_fail_stats[type][status]);
    }
  }
}

void CacheStats ::print_energy_stats(FILE *out, const char *cache_name) const {
  fprintf(out, "%s_RH: %lu\n", cache_name, get_read_hit());
  fprintf(out, "%s_RM: %lu\n", cache_name, get_read_miss());
  fprintf(out, "%s_WH: %lu\n", cache_name, get_write_hit());
  fprintf(out, "%s_WM: %lu\n", cache_name, get_write_miss());
}

bool CacheStats::check_valid(int access_type, int access_outcome) const {
  return (access_type >= 0 && access_type < NUM_MEM_ACCESS_TYPE &&
          access_outcome >= 0 && access_outcome < NUM_CACHE_REQUEST_STATUS);
}

bool CacheStats::check_fail_valid(int access_type, int fail_outcome) const {
  return (access_type >= 0 && access_type < NUM_MEM_ACCESS_TYPE &&
          fail_outcome >= 0 &&
          fail_outcome < NUM_CACHE_RESERVATION_FAIL_REASON);
}
