#pragma once

#include <cstdint>
#include <string>

#include <spdlog/spdlog.h>

#include "Instruction.h"
#include "TraceLogTags.h"

/**
 * Instruction / tile trace formatting and Core spdlog::trace helpers.
 * Keeps Core.cc focused on simulation logic.
 */
namespace core_trace_log {

/** Whether the instruction trace is actually being emitted.
 *
 * The trace_* helpers below take formatted strings, so their arguments are
 * built whether or not spdlog will keep them: every issued instruction paid two
 * std::string constructions (fmt::format over its tile dims/strides) that
 * spdlog::trace then dropped. Guard the call sites with this. */
inline bool trace_enabled() { return spdlog::should_log(spdlog::level::trace); }

std::string format_dma_inst_issued_detail(Instruction& inst);
/** Opcode + (detail...) for DMA issue / skip traces. */
std::string format_dma_inst_issued_trace_line(Instruction& inst);
/** Opcode + (detail...) for COMP / BAR / MOVIN / MOVOUT finished or issued lines. */
std::string format_instruction_detail_line(Instruction& inst);

void trace_tile_scheduled(cycle_type core_cycle, uint32_t core_id, const std::string& tag15,
                          size_t spad_footprint, int max_dispatch);

void trace_instruction_line(cycle_type core_cycle,
                            uint32_t core_id,
                            const std::string& tag15,
                            uint64_t global_inst_id,
                            const std::string& message);

void log_error_dma_instruction_invalid(cycle_type core_cycle, uint32_t core_id);
void log_error_dram_responses_trace_not_finished(cycle_type core_cycle, uint32_t core_id);
void log_error_instruction_already_finished(cycle_type core_cycle,
                                            uint32_t core_id,
                                            const std::string& opcode_name);
void log_error_undefined_opcode();

}  // namespace core_trace_log
