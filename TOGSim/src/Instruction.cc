#include "Instruction.h"

#include <fmt/format.h>

uint64_t Instruction::_next_global_inst_id = 0;

std::string format_tag_key_list_hex(const std::vector<int64_t>& tag_keys) {
  if (tag_keys.empty()) {
    return {};
  }
  std::string out;
  for (size_t i = 0; i < tag_keys.size(); ++i) {
    if (i > 0) {
      out.push_back(',');
    }
    out += fmt::format("0x{:016x}", static_cast<uint64_t>(tag_keys[i]));
  }
  return out;
}

std::string opcode_to_string(Opcode opcode) {
    switch (opcode) {
        case Opcode::MOVIN:        return "MOVIN";
        case Opcode::MOVOUT:       return "MOVOUT";
        case Opcode::COMP:         return "COMP";
        case Opcode::MEMORY_BAR:   return "MEMORY_BAR";
        case Opcode::COMPUTE_BAR:  return "COMPUTE_BAR";
        default:                   return "Unknown";
    }
}

Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, std::vector<int> tile_stride, size_t elem_bits,
            std::vector<int64_t> tag_idx_list, std::vector<int64_t> tag_stride_list,
            std::vector<int64_t> accum_tag_idx_list)
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(tile_size), tile_stride(tile_stride), _elem_bits(elem_bits),
    _tag_idx_list(tag_idx_list), _tag_stride_list(tag_stride_list),
    _accum_tag_idx_list(accum_tag_idx_list) {
  _global_inst_id = _next_global_inst_id++;
  assert(_tag_idx_list.size()==_tag_stride_list.size());
  _tile_numel = 1;
  for (auto dim : tile_size)
    _tile_numel *= dim;
}

Instruction::Instruction(Opcode opcode)
  : opcode(opcode) {
  _global_inst_id = _next_global_inst_id++;
  _tile_numel = 1;
}

void Instruction::finish_instruction() {
  for (auto& counter : child_inst)
    counter->dec_ready_counter();
  finished = true;
}

void Instruction::add_child(std::shared_ptr<Instruction> child) {
  child->inc_ready_counter();
  child_inst.insert(child);
}

void Instruction::add_pipeline_child(std::shared_ptr<Instruction> child) {
  child->inc_ready_counter();
  _pipeline_children.insert(child);
}

void Instruction::release_pipeline_children() {
  for (auto& c : _pipeline_children) {
    // a COMPUTE_BAR child fences only its own dispatch -> it drains the max
    // finish of the computes it gates, fed here as each one issues.
    if (c->get_opcode() == Opcode::COMPUTE_BAR) c->update_fence_finish(finish_cycle);
    c->dec_ready_counter();
  }
  _pipeline_children.clear();
}

void Instruction::inc_waiting_request() {
  _nr_waiting_request++;
}

void Instruction::dec_waiting_request() {
  assert(_nr_waiting_request!=0);
  _nr_waiting_request--;
}

void Instruction::prepare_tag_key() {
  /* Calculate tag key */
  int64_t key_offset = 0;
  _tag_key.push_back(_addr_id);
  for (size_t i = 0; i < _tag_idx_list.size(); i++)
    key_offset += _tag_idx_list.at(i) * _tag_stride_list.at(i);
  for (auto accum_dim : _accum_tag_idx_list)
    _tag_key.push_back(accum_dim);
  _tag_key.push_back(key_offset);
}

void Instruction::print() {
  spdlog::info("{}", opcode_to_string(opcode));
}

std::shared_ptr<std::set<addr_type>> Instruction::get_dram_address(addr_type dram_req_size) {
  auto address_set = std::make_shared<std::set<addr_type>>();
  uint64_t* indirect_index = NULL;
  size_t index_count = 0;
  /* Set 4D shape*/
  while (tile_size.size() < 4)
    tile_size.insert(tile_size.begin(), 1);

  while (tile_stride.size() < 4)
    tile_stride.insert(tile_stride.begin(), 0);
  if (_is_indirect_mode) {
    spdlog::trace("[Indirect Access] Indirect mode, dump_path: {}", _indirect_index_path);
    load_indirect_index(_indirect_index_path, indirect_index, tile_size);
  }

  /* Iterate tile_size */
  for (int dim0=0; dim0<tile_size.at(0); dim0++) {
    for (int dim1=0; dim1<tile_size.at(1); dim1++) {
      for (int dim2=0; dim2<tile_size.at(2); dim2++) {
        for (int dim3=0; dim3<tile_size.at(3); dim3++) {
          addr_type address = dim0*tile_stride.at(tile_stride.size() - 4) + \
                              dim1*tile_stride.at(tile_stride.size() - 3) + \
                              dim2*tile_stride.at(tile_stride.size() - 2) + \
                              dim3*tile_stride.at(tile_stride.size() - 1);
          address = dram_addr + (address * _elem_bits + 7) >> 3;
          if (indirect_index != NULL) {
            uint64_t index_val = indirect_index[index_count++];
            address += (index_val * _elem_bits + 7) >> 3;
          }
          address_set->insert(address - (address & dram_req_size-1));
        }
      }
    }
  }
  return address_set;
}

bool Instruction::load_indirect_index(const std::string& path, uint64_t*& indirect_index, const std::vector<uint64_t>& tile_size) {
  size_t count;
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    spdlog::warn("[Indirect Access] Failed to open index file(\'{}\')", path);
    return false;
  }

  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  count = size / sizeof(uint64_t);

  uint64_t expected_count = tile_size[0] * tile_size[1] * tile_size[2] * tile_size[3];
  if (size % sizeof(uint64_t) != 0 || count != expected_count) {
    spdlog::warn("[Indirect Access] Invalid file size ({} Bytes) at \'{}\'", size, path);
    return false;
  }

  indirect_index = new uint64_t[count];

  if (!ifs.read(reinterpret_cast<char*>(indirect_index), size)) {
    spdlog::warn("[Indirect Access] Failed to read data from file (\'{}\')", path);
    delete[] indirect_index;
    indirect_index = NULL;
    count = 0;
    return false;
  }
  return true;
}