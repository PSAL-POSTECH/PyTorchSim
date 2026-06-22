#pragma once

#include <string>
#include <string_view>

/** Trace bracket tags: max 15 characters; use pad15() so logs show a fixed 15-char field (space-padded). */
namespace TraceLogTag {

/** Right-pad (or truncate) to exactly 15 characters for aligned log columns. */
inline std::string pad15(std::string_view sv) {
  if (sv.size() > 15) {
    sv = sv.substr(0, 15);
  }
  std::string out(sv);
  out.resize(15, ' ');
  return out;
}

inline constexpr const char* kTileScheduled = "TILE_SCHEDULED";

inline constexpr const char* kInstructionIssued = "INST_ISSUED";
inline constexpr const char* kInstructionFinished = "INST_FINISHED";
/** Async MOVIN skipped: same tag still in flight. */
inline constexpr const char* kInstructionSkipped = "INST_SKIP";

inline constexpr const char* kAsyncDmaAllRequestsIssued = "ASYNC_DMA_ISSUE";
inline constexpr const char* kFirstDramResponse = "DRAM_RESP_FIRST";
inline constexpr const char* kAllDramResponsesReceived = "DRAM_RESP_DONE";

inline constexpr const char* kL2CacheableStatusForAddress = "L2CACHE_STAT";
inline constexpr const char* kDmaNumaPlacement = "DRAM_NUMA";

/** Field label for get_global_inst_id() in trace lines (≤15 chars). */
inline constexpr const char* kGlobalInstIdKey = "INST_ID";
}  // namespace TraceLogTag
