#ifndef CACHE_DEFS_H
#define CACHE_DEFS_H
#include <bitset>
#include <list>
#include <map>
#include <cstdint>
#include <deque>

const int SECTOR_CHUNCK_SIZE = 4;
typedef std::bitset<SECTOR_CHUNCK_SIZE> SectorMask;
enum CacheBlockState {
  INVALID,  // Initial state
  RESERVED, // Reserved state (alloc())
  VALID,    // Filled state (fill)
  MODIFIED  // Filled with modified data (fill)
};
enum CacheRequestStatus {
  HIT,
  HIT_RESERVED,
  MISS,
  RESERVATION_FAIL,
  SECTOR_MISS,
  MSHR_HIT,
  NUM_CACHE_REQUEST_STATUS
};
static const char *cache_request_status_str[] = {
    "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
    "SECTOR_MISS", "MSHR_HIT"};

enum CacheReservationFailReason {
  LINE_ALLOC_FAIL,
  MISS_QUEUE_FULL,
  MSHR_ENTRY_FAIL,
  MSHR_MERGE_ENTRY_FAIL,
  MSHR_RW_PENDING,
  NUM_CACHE_RESERVATION_FAIL_REASON
};
static const char *cache_reservation_fail_reason_str[] = {
    "LINE_ALLOCATE_FAIL", "MISS_QUEUE_FULL", "MSHR_ENTRY_FAIL",
    "MSHR_MERGE_ENTRY_FAIL", "MSHR_RW_PENDING"};

enum CacheEventType {
  WRITE_BACK_REQUEST_SENT,
  READ_REQUEST_SENT,
  WRITE_REQUEST_SENT,
  WRITE_ALLOCATE_SENT
};

struct EvictedBlockInfo {
  uint64_t m_block_addr = 0;
  uint32_t m_modified_size = 0;
  SectorMask m_dirty_mask;
  void set_info(uint64_t block_addr, uint32_t modified_size,
                SectorMask dirty_mask) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
    m_dirty_mask = dirty_mask;
  }
};
struct CacheEvent {
  CacheEvent() {}
  CacheEvent(CacheEventType cache_event_type)
      : m_cache_event_type(cache_event_type) {}
  CacheEvent(CacheEventType cache_event_type, EvictedBlockInfo evicted_block)
      : m_cache_event_type(cache_event_type), m_evicted_block(evicted_block) {}
  CacheEventType m_cache_event_type;
  EvictedBlockInfo m_evicted_block;  // only valid for WRITE_BACK_REQUEST_SENT
  static bool was_event_sent(const std::deque<CacheEvent> &events,
                             CacheEventType event_type,
                             CacheEvent &found_event) {
    for (auto &event : events) {
      if (event.m_cache_event_type == event_type) {
        found_event = event;
        return true;
      }
    }
    return false;
  }
  static bool was_write_sent(const std::deque<CacheEvent> &events) {
    CacheEvent event;
    return was_event_sent(events, WRITE_REQUEST_SENT, event);
  }
  static bool was_read_sent(const std::deque<CacheEvent> &events) {
    CacheEvent event;
    return was_event_sent(events, READ_REQUEST_SENT, event);
  }
  static bool was_writeback_sent(const std::deque<CacheEvent> &events,
                                 CacheEvent event) {
    return was_event_sent(events, WRITE_BACK_REQUEST_SENT, event);
  }
  static bool was_write_allocate_sent(const std::deque<CacheEvent> &events) {
    CacheEvent event;
    return was_event_sent(events, WRITE_ALLOCATE_SENT, event);
  }
};

enum WritePolicy {
  READ_ONLY,
  WRITE_BACK,
  WRITE_THROUGH,
  WRITE_EVICT,
  LOCAL_WB_GLOBAL_WT
};
static std::map<char, WritePolicy> WritePolicyMap = {{'R', READ_ONLY},
                                                     {'B', WRITE_BACK},
                                                     {'T', WRITE_THROUGH},
                                                     {'E', WRITE_EVICT},
                                                     {'L', LOCAL_WB_GLOBAL_WT}};

enum AllocationPolicy { ON_MISS, ON_FILL, STREAMING };
static std::map<char, AllocationPolicy> AllocationPolicyMap = {
    {'m', ON_MISS}, {'f', ON_FILL}, {'s', STREAMING}};

enum WriteAllocatePolicy {
  NO_WRITE_ALLOCATE,
  WRITE_ALLOCATE,
  FETCH_ON_WRITE,
  LAZY_FETCH_ON_READ
};
static std::map<char, WriteAllocatePolicy> WriteAllocatePolicyMap = {
    {'N', NO_WRITE_ALLOCATE},
    {'W', WRITE_ALLOCATE},
    {'F', FETCH_ON_WRITE},
    {'L', LAZY_FETCH_ON_READ}};

enum CacheType { NORMAL, SECTOR };
static std::map<char, CacheType> CacheTypeMap = {{'N', NORMAL}, {'S', SECTOR}};

enum EvictPolicy { LRU, FIFO };
static std::map<char, EvictPolicy> EvictPolicyMap = {{'L', LRU}, {'F', FIFO}};

enum MshrConfig { ASSOC, SECTOR_ASSOC };
static std::map<char, MshrConfig> MshrConfigMap = {{'A', ASSOC},
                                                   {'S', SECTOR_ASSOC}};

enum SetIndexFunction {
  LINEAR_SET_FUNCTION,
  BITWISE_XORING_FUNCTION,
  HASH_IPOLY_FUNCTION,
  CUSTOM_SET_FUNCTION
};
static std::map<char, SetIndexFunction> SetIndexFunctionMap = {
    {'L', LINEAR_SET_FUNCTION},
    {'X', BITWISE_XORING_FUNCTION},
    {'P', HASH_IPOLY_FUNCTION},
    {'C', CUSTOM_SET_FUNCTION}};
#endif