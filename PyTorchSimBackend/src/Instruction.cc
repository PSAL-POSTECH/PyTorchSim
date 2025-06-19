#include "Instruction.h"

std::string opcode_to_string(Opcode opcode) {
    switch (opcode) {
        case Opcode::MOVIN:        return "MOVIN";
        case Opcode::MOVOUT:       return "MOVOUT";
        case Opcode::COMP:         return "COMP";
        case Opcode::BAR:          return "BAR";
        default:                   return "Unknown";
    }
}

Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, size_t precision,
            std::vector<int>& idx_list, std::vector<int>& stride_list,
            std::vector<int> tag_idx_list, std::vector<int> tag_stride_list,
            std::vector<int> accum_tag_idx_list, std::vector<int> loop_size_list)
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(tile_size), _precision(precision), _idx_list(idx_list),
    _stride_list(stride_list), _tag_idx_list(tag_idx_list), _tag_stride_list(tag_stride_list),
    _accum_tag_idx_list(accum_tag_idx_list), _loop_size_list(loop_size_list) {
  assert(_tag_idx_list.size()==_tag_stride_list.size());
  _tile_numel = 1;
  for (auto dim : tile_size)
    _tile_numel *= dim;

  /* Supporting vector */
  if (_stride_list.size() == 1) {
    _stride_list.push_back(1);
  }
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

void Instruction::inc_waiting_request() {
  _nr_waiting_request++;
}

void Instruction::dec_waiting_request() {
  assert(_nr_waiting_request!=0);
  _nr_waiting_request--;
}

void Instruction::prepare_tag_key() {
  /* Calculate tag key */
  int key_offset = 0;
  _tag_key.push_back(_addr_id);
  for (int i=0; i<_tag_idx_list.size(); i++)
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

  while (_stride_list.size() < 4)
    _stride_list.insert(_stride_list.begin(), 0);
  if (_is_indirect_mode) {
    spdlog::trace("[Indirect Access] Indirect mode, dump_path: {}", _indirect_index_path);
    load_indirect_index(_indirect_index_path, indirect_index, tile_size);
  }

  /* Iterate tile_size */
  for (int dim0=0; dim0<tile_size.at(0); dim0++) {
    for (int dim1=0; dim1<tile_size.at(1); dim1++) {
      for (int dim2=0; dim2<tile_size.at(2); dim2++) {
        for (int dim3=0; dim3<tile_size.at(3); dim3++) {
          addr_type address = dim0*_stride_list.at(_stride_list.size() - 4) + \
                              dim1*_stride_list.at(_stride_list.size() - 3) + \
                              dim2*_stride_list.at(_stride_list.size() - 2) + \
                              dim3*_stride_list.at(_stride_list.size() - 1);
          address = dram_addr + address * _precision;
          if (indirect_index != NULL) {
            uint64_t index_val = indirect_index[index_count++];
            address += index_val * _precision;
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