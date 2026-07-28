#ifndef CACHE_STATS_H
#define CACHE_STATS_H
#include <vector>
#include <cstdint>
#include <cassert>

#include "Cache_defs.h"

class CacheStats {
 public:
  CacheStats();
  void clear();
  void inc_stats(int access_type, int accss_outcome);
  void inc_fail_stats(int access_type, int fail_outcome);
  CacheRequestStatus select_stats_status(CacheRequestStatus probe,
                                         CacheRequestStatus access) const;
  uint64_t &operator()(int access_type, int access_outcome, bool fail_outcome);
  uint64_t operator()(int access_type, int access_outcome,
                      bool fail_outcome) const;
  CacheStats operator+(const CacheStats &other);
  CacheStats &operator+=(const CacheStats &other);
  void aggregate_stats();
  uint64_t get_hit() const;
  uint64_t get_read_hit() const;
  uint64_t get_write_hit() const;
  uint64_t get_miss() const;
  uint64_t get_read_miss() const;
  uint64_t get_write_miss() const;
  uint64_t get_accesses() const;
  uint64_t get_interval_hit();
  uint64_t get_interval_miss();
  void print_stats(FILE *out, const char *cache_name = "CacheStats") const;
  void print_fail_stats(FILE *out, const char *cache_name = "CacheStats") const;
  void print_energy_stats(FILE *out,
                          const char *cache_name = "CacheStats") const;

 private:
  bool check_valid(int type, int status) const;
  bool check_fail_valid(int type, int fail) const;

  std::vector<std::vector<uint64_t>> m_stats;
  std::vector<std::vector<uint64_t>> m_fail_stats;

  uint64_t m_cache_port_available_cycles;
  uint64_t m_cache_data_port_busy_cycles;
  uint64_t m_cache_fill_port_busy_cycles;

  uint64_t m_prev_hit;
  uint64_t m_prev_miss;
};
#endif