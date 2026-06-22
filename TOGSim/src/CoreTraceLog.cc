#include "CoreTraceLog.h"

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

namespace core_trace_log {

std::string format_dma_inst_issued_detail(Instruction& inst) {
  const auto& ts = inst.get_tile_size();
  const int rank = static_cast<int>(std::max<size_t>(1, ts.size()));
  if (inst.get_opcode() == Opcode::MOVIN) {
    return fmt::format(
        "addr_name={} dram=0x{:016x} rank={} size=[{}] stride=[{}] elem_bits={} async={} indirect={} tag_id=[{}]",
        inst.get_addr_name(),
        static_cast<uint64_t>(inst.get_base_dram_address()),
        rank,
        fmt::join(ts, ","),
        fmt::join(inst.get_tile_stride(), ","),
        inst.get_elem_bits(),
        inst.is_async_dma(),
        inst.is_indirect_mode(),
        format_tag_key_list_hex(inst.get_tag_id()));
  }
  uint64_t tag_hex = 0;
  const auto& tidx = inst.get_tag_idx_list();
  if (!tidx.empty()) {
    tag_hex = static_cast<uint64_t>(tidx[0]);
  }
  return fmt::format(
      "addr_name={} dram=0x{:016x} rank={} elem_bits={} async={} indirect={} tag=0x{:016x} stride=[{}] size=[{}] "
      "tag_idx=[{}] tile={}",
      inst.get_addr_name(),
      static_cast<uint64_t>(inst.get_base_dram_address()),
      rank,
      inst.get_elem_bits(),
      inst.is_async_dma(),
      inst.is_indirect_mode(),
      tag_hex,
      fmt::join(inst.get_tile_stride(), ","),
      fmt::join(ts, ","),
      fmt::join(tidx, ","),
      inst.get_tile_group());
}

std::string format_dma_inst_issued_trace_line(Instruction& inst) {
  return fmt::format("{} ({})", opcode_to_string(inst.get_opcode()), format_dma_inst_issued_detail(inst));
}

std::string format_instruction_detail_line(Instruction& inst) {
  const Opcode op = inst.get_opcode();
  const std::string opname = opcode_to_string(op);
  if (op == Opcode::COMP) {
    return fmt::format("{} (compute_type={} compute_cycle={} overlapping_cycle={} sa={} tile={})",
                       opname,
                       inst.get_compute_type(),
                       inst.get_compute_cycle(),
                       inst.get_overlapping_cycle(),
                       inst.get_assigned_sa(),
                       inst.get_tile_group());
  }
  if ((op == Opcode::MOVIN || op == Opcode::MOVOUT) && inst.is_async_dma()) {
    return fmt::format("{} (ASYNC subgraph_id={} addr_name={} tag_id=[{}] tag_idx=[{}] tag_stride=[{}] tile={})",
                       opname,
                       inst.subgraph_id,
                       inst.get_addr_name(),
                       format_tag_key_list_hex(inst.get_tag_id()),
                       fmt::join(inst.get_tag_idx_list(), ","),
                       fmt::join(inst.get_tag_stride_list(), ","),
                       inst.get_tile_group());
  }
  if (op == Opcode::MOVIN || op == Opcode::MOVOUT) {
    return fmt::format("{} (addr_name={} tile={})", opname, inst.get_addr_name(), inst.get_tile_group());
  }
  if (op == Opcode::MEMORY_BAR) {
    return fmt::format("{} (addr_name={} tag_id=[{}] tag_idx=[{}] tag_stride=[{}] tile={})",
                       opname,
                       inst.get_addr_name(),
                       format_tag_key_list_hex(inst.get_tag_id()),
                       fmt::join(inst.get_tag_idx_list(), ","),
                       fmt::join(inst.get_tag_stride_list(), ","),
                       inst.get_tile_group());
  }
  return opname;
}

void trace_tile_scheduled(cycle_type core_cycle, uint32_t core_id, const std::string& tag15) {
  spdlog::trace("[{}][Core {}][{}]", core_cycle, core_id, tag15);
}

void trace_instruction_line(cycle_type core_cycle,
                            uint32_t core_id,
                            const std::string& tag15,
                            uint64_t global_inst_id,
                            const std::string& message) {
  spdlog::trace("[{}][Core {}][{}][{}={}] {}",
                 core_cycle,
                 core_id,
                 tag15,
                 TraceLogTag::kGlobalInstIdKey,
                 global_inst_id,
                 message);
}

void log_error_dma_instruction_invalid(cycle_type core_cycle, uint32_t core_id) {
  spdlog::error("[{}][Core {}] DMA instruction in not valid", core_cycle, core_id);
}

void log_error_dram_responses_trace_not_finished(cycle_type core_cycle, uint32_t core_id) {
  spdlog::error("[{}][Core {}][ERROR] ALL_DRAM_RESPONSES_RECEIVED trace but inst not finished yet",
                core_cycle,
                core_id);
}

void log_error_instruction_already_finished(cycle_type core_cycle,
                                            uint32_t core_id,
                                            const std::string& opcode_name) {
  spdlog::error("[{}][Core {}][ERROR] {} inst already finished!!", core_cycle, core_id, opcode_name);
}

void log_error_undefined_opcode() {
  spdlog::error("Undefined instruction opcode type");
}

}  // namespace core_trace_log
