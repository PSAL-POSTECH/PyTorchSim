#pragma once
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <fmt/ranges.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "TileGraph.h"
#include "Instruction.h"
#include "sstStonne.h"
#include "IntervalTree.h"
#include "Common.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

enum class TileType{
  LOOP_INDEX_NODE,
  LOOP_END_NODE,
  LOAD_NODE,
  STORE_NODE,
  COMPUTE_NODE,
  MEMORY_WAIT_NODE,
  STONNE_NODE,
  STONNE_TRACE_COMPUTE_NODE,
  STONNE_TRACE_LOAD_NODE,
  STONNE_TRACE_STORE_NODE
};

enum class LoopType {
  NORMAL_LOOP,
  PARALLEL_LOOP,
  ACCUMULATION_LOOP,
  INNER_LOOP
};

class TileNode {
 public:
  TileNode(onnx::NodeProto& node);
  static TileType get_tile_type(std::string type);
  void add_child(std::shared_ptr<TileNode> child) { _child.push_back(std::move(child)); }
  std::vector<std::shared_ptr<TileNode>>& get_child() { return _child; }
  void add_parent(std::shared_ptr<TileNode> parent) { _parent.push_back(std::move(parent)); }
  std::vector<std::shared_ptr<TileNode>>& get_parent() { return _parent; }
  std::vector<std::string>& get_child_name() { return _child_name; }
  std::vector<std::string>& get_parent_name() { return _parent_name; }
  TileType get_type() { return _type; }
  std::shared_ptr<TileNode> get_owner_loop() { return _owner_loop; }
  std::string get_name() { return _name; }
  void set_owner_loop(std::shared_ptr<TileNode> owner) { _owner_loop=std::move(owner); }
  virtual void print_node();
  void set_depth(int depth) { _depth=depth; }
  int get_depth() { return _depth; }

 private:
  std::vector<std::shared_ptr<TileNode>> _parent;
  std::vector<std::shared_ptr<TileNode>> _child;
  std::vector<std::string> _parent_name;
  std::vector<std::string> _child_name;
  std::shared_ptr<TileNode> _owner_loop;
  std::string _name;
  int _depth;
  TileType _type;
};

class TileGraphParser {
 public:
  TileGraphParser(std::string onnx_path, std::string attribute_path, const YAML::Node& config_yaml);
  std::shared_ptr<TileNode> get_top_loop();
  std::unique_ptr<TileGraph>& get_tile_graph() { return _tile_graph; }
  addr_type lookup(std::string key);
  void register_loop(std::shared_ptr<TileNode>);
  void increase_loop_top() { _loop_stack_pointer++; }
  void decrease_loop_top() { _loop_stack_pointer--; }
  int get_loop_size(std::string key) { return std::get<0>(_loop_size_map[key]); }
  int get_loop_step(std::string key) { return std::get<1>(_loop_size_map[key]); }
  LoopType get_loop_type(std::string key) { return std::get<2>(_loop_size_map[key]); }
  const std::map<std::string, std::tuple<int, int, LoopType>> & get_loop_map() { return _loop_size_map; }
  const std::vector<uint32_t> &lookupNumaInfo(std::string key);
  int getCoreIdFromConfig(const YAML::Node& attribute_config, int subgraph_id);
  std::string getMetaByName(std::string key) { return _tog_meta[key]; }
  const YAML::Node& get_attribute_file() { return _attribute_config; }
  std::vector<int64_t> calc_tag(std::vector<int64_t>& accum_tag, std::vector<int64_t>& tag_idx, std::vector<int64_t>& tag_stride);
  void register_memory_tag(std::string name, std::vector<int64_t>& tag_key);
  bool check_memory_tag(std::string name, std::vector<int64_t>& tag_key);
  void clear_tag_table() { _tag_table.clear(); }
  std::string get_indirect_path() {
    namespace fs = std::filesystem;
    fs::path original(_attribute_path);
    fs::path base_folder = original.parent_path().parent_path();
    fs::path new_path = base_folder / "indirect_access" / (std::string("indirect_index") + std::to_string(indirect_counter) + ".raw");
    return new_path.string();
  }
  std::string get_sparse_tile_meta_path() {
    namespace fs = std::filesystem;
    fs::path original(_attribute_path);
    fs::path base_folder = original.parent_path().parent_path();
    fs::path new_path = base_folder / "dma_access" / (std::string("sparse_tile.raw"));
    return new_path.string();
  }
  void load_sparse_meta_data() {
    /* Prepare runtime attribute */
    std::string sparse_meta_path = get_sparse_tile_meta_path();
    std::ifstream file(sparse_meta_path, std::ios::binary);
    if (file) {
      file.seekg(0, std::ios::end);
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);
      size_t count = size / sizeof(int64_t);
      for (size_t i = 0; i < count; ++i) {
          int64_t val;
          file.read(reinterpret_cast<char*>(&val), sizeof(int64_t));
          sparse_tile_set.insert(val);
      }
    }
  }
  void inc_indirect_counter() { indirect_counter++; }
  uint64_t get_dma_counter() { return dma_counter; }
  void inc_dma_counter() { dma_counter++; }
  bool is_sparse_tile(uint64_t idx) { return sparse_tile_set.find(idx) != sparse_tile_set.end(); }
  int64_t register_addr_name(const std::string& addr_name) {
    if (_addr_name_map.find(addr_name) == _addr_name_map.end())
      _addr_name_map[addr_name] = static_cast<int64_t>(_addr_name_map.size());
    return _addr_name_map[addr_name];
  }
  int64_t get_addr_name_id(const std::string& addr_name) { return _addr_name_map[addr_name]; }

 private:
  void register_tile(std::shared_ptr<TileNode> tile_node);
  void _tile_generate() {}
  void _base_addr_update() {}
  void _tile_index_generate() {}
  int _loop_stack_pointer = 0;

  YAML::Node _attribute_config; 
  YAML::Node _config_yaml;
  std::string _tog_path;
  std::string _attribute_path;
  uint64_t indirect_counter = 0;
  uint64_t dma_counter = 0;
  std::set<uint64_t> sparse_tile_set;
  std::map<std::string, std::shared_ptr<TileNode>> _output_map;
  std::vector<std::vector<std::shared_ptr<TileNode>>> _loop_nodes;
  std::vector<std::shared_ptr<TileNode>> _tile_vec;
  std::unique_ptr<TileGraph> _tile_graph;
  std::map<std::string, addr_type> _arg_to_address;
  std::map<std::string, std::vector<uint32_t>> _arg_numa_stride;
  std::vector<Interval<unsigned long long, int>> _cache_plan;
  std::map<std::string, std::tuple<int, int, LoopType>> _loop_size_map;
  std::map<std::string, std::string> _tog_meta;
  std::map<std::pair<std::string, std::vector<int64_t>>, uint32_t> _tag_table;
  std::unordered_map<std::string, int64_t> _addr_name_map;
};

class TileComputeNode : public TileNode {
 public:
  TileComputeNode(onnx::NodeProto& node);
  uint32_t get_cycle() { return _cycle; }
  uint32_t get_overlapping_cycle() { return _overlapping_cycle; }
  int get_compute_type() { return _compute_type; }
  void print_node();

 private:
  std::map<std::string, std::shared_ptr<TileNode>> tile_map;
  uint32_t _cycle;
  uint32_t _overlapping_cycle = 0;
  int _compute_type;
};

class TileMemoryNode : public TileNode {
 public:
  TileMemoryNode(onnx::NodeProto& node);
  std::string get_base_addr_name() { return _base_addr_name; }
  size_t get_elem_bits() const { return _elem_bits; }
  std::vector<size_t> get_tile_size() { return _tile_size; }
  std::vector<int>& get_tile_stride() { return _tile_stride; }
  std::vector<std::string>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<int64_t>& get_tag_stride_list() { return _tag_stride_list; }
  std::vector<std::string>& get_loop_idx_list() { return _loop_idx_list; }
  std::vector<int>& get_loop_stride_list () { return _loop_stride_list; }
  bool is_async_node() { return _is_async; }
  bool is_indirect() { return _is_indirect; }
  void print_node() override;

 private:
  std::vector<size_t> _tile_size;
  std::vector<int> _tile_stride;
  size_t _elem_bits = 0;
  bool _is_async;
  bool _is_indirect;
  std::string _base_addr_name;
  std::vector<std::string> _tag_idx_list;
  std::vector<int64_t> _tag_stride_list;
  std::vector<std::string> _loop_idx_list;
  std::vector<int> _loop_stride_list;
};

class TileMemoryWaitNode : public TileNode {
 public:
  TileMemoryWaitNode(onnx::NodeProto& node);
  std::string get_base_addr_name() { return _base_addr_name; }
  std::vector<std::string>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<int64_t>& get_tag_stride_list() { return _tag_stride_list; }
  std::vector<int64_t>& get_tag_divider_list() { return _tag_divider_list; }
  void print_node() override;

 private:
  std::vector<std::string> _tag_idx_list;
  std::vector<int64_t> _tag_stride_list;
  std::vector<int64_t> _tag_divider_list;
  std::string _base_addr_name;
};

class TileLoopNode : public TileNode {
 public:
 TileLoopNode(onnx::NodeProto& node);
  void add_body(std::shared_ptr<TileNode> body) { _body_node.push_back(body); }
  std::vector<std::shared_ptr<Tile>> get_tiles_from_iter(TileGraphParser*, std::map<std::string, int>&);
  std::string get_idx_name() { return _tile_index_name; }
  uint64_t get_start() { return _start; }
  uint64_t get_stride() { return _stride; }
  uint64_t get_end() { return _end; }
  LoopType get_loop_type() { return _loop_type; }
  void print_node() override;
 private:
  std::string _tile_index_name;
  uint64_t _stride;
  uint64_t _start;
  uint64_t _end;
  LoopType _loop_type;
  std::vector<std::shared_ptr<TileNode>> _body_node;
};

class TileLoopEndNode : public TileNode {
 public:
  TileLoopEndNode(onnx::NodeProto& node) : TileNode(node) {}
};

class TileStonneNode : public TileNode {
 public:
  TileStonneNode(onnx::NodeProto& node) : TileNode(node) {
    for (auto attribute : node.attribute()) {
      if (attribute.name() == "torchsim_stonne_operation") {
        std::string op_type = attribute.s();
        if (op_type == "CONV") {
            desc.operation = Layer_t::CONV;
        } else if (op_type == "GEMM") {
            desc.operation = Layer_t::GEMM;
        } else if (op_type == "POOL") {
            desc.operation = Layer_t::POOL;
        } else if (op_type == "FC") {
            desc.operation = Layer_t::FC;
        } else if (op_type == "SPARSE_DENSE") {
            desc.operation = Layer_t::SPARSE_DENSE;
        } else if (op_type == "bitmapSpMSpM") {
            desc.operation = Layer_t::bitmapSpMSpM;
        } else if (op_type == "csrSpMM") {
            desc.operation = Layer_t::csrSpMM;
        } else if (op_type == "outerProductGEMM") {
            desc.operation = Layer_t::outerProductGEMM;
        } else if (op_type == "gustavsonsGEMM") {
            desc.operation = Layer_t::gustavsonsGEMM;
        } else {
            spdlog::error("[TileStonneNode] Unknown operation type: {}", op_type);
            throw std::runtime_error("Invalid operation type in TileStonneNode");
        }
      } else if (attribute.name() == "torchsim_stonne_layer_name") {
          desc.layer_name = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_mem_init") {
          desc.mem_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_R") {
          desc.R = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_S") {
          desc.S = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_C") {
          desc.C = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_K") {
          desc.K = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_G") {
          desc.G = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_N") {
          desc.N = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_X") {
          desc.X = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_Y") {
          desc.Y = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_X_") {
          desc.X_ = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_Y_") {
          desc.Y_ = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_strides") {
          desc.strides = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_R") {
          desc.T_R = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_S") {
          desc.T_S = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_C") {
          desc.T_C = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_K") {
          desc.T_K = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_G") {
          desc.T_G = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_N") {
          desc.T_N = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_X_") {
          desc.T_X_ = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_T_Y_") {
          desc.T_Y_ = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_K") {
          desc.GEMM_K = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_N") {
          desc.GEMM_N = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_M") {
          desc.GEMM_M = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_T_K") {
          desc.GEMM_T_K = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_T_N") {
          desc.GEMM_T_N = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_GEMM_T_M") {
          desc.GEMM_T_M = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_matrix_a_dram_address") {
          desc.matrix_a_dram_address = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_matrix_b_dram_address") {
          desc.matrix_b_dram_address = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_matrix_c_dram_address") {
          desc.matrix_c_dram_address = attribute.i();
      } else if (attribute.name() == "torchsim_stonne_mem_matrix_c_file_name") {
          desc.mem_matrix_c_file_name = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_bitmap_matrix_a_init") {
          desc.bitmap_matrix_a_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_bitmap_matrix_b_init") {
          desc.bitmap_matrix_b_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_rowpointer_matrix_a_init") {
          desc.rowpointer_matrix_a_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_colpointer_matrix_a_init") {
          desc.colpointer_matrix_a_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_rowpointer_matrix_b_init") {
          desc.rowpointer_matrix_b_init = attribute.s();
      } else if (attribute.name() == "torchsim_stonne_colpointer_matrix_b_init") {
          desc.colpointer_matrix_b_init = attribute.s();
      } else if (attribute.name() == "torchsim_bitmap_matrix_a_init") {
          desc.bitmap_matrix_a_init = attribute.s();
      } else if (attribute.name() == "torchsim_bitmap_matrix_b_init") {
          desc.bitmap_matrix_b_init = attribute.s();
      }  else if (attribute.name() == "torchsim_mem_matrix_c_file_name") {
          desc.mem_matrix_c_file_name = attribute.s();
      }  else if (attribute.name() == "torchsim_trace_path") {
          desc.trace_path = attribute.s();
      } else {
          spdlog::warn("[TileStonneNode] Unrecognized attribute: {}", attribute.name());
      }
    }
  }
  SST_STONNE::StonneOpDesc* getDesc() { return &desc; }
  void print_node() override;
 private:
  SST_STONNE::StonneOpDesc desc;
};

class TileStonneTraceComputeNode : public TileNode {
 public:
  TileStonneTraceComputeNode(onnx::NodeProto& node) : TileNode(node) {
    for (auto attribute : node.attribute()) {
      if (attribute.name() == "torchsim_trace_compute_cycle") {
          _cycle = attribute.i();
      }
    }
  }
  uint32_t get_cycle() { return _cycle; }
  void print_node();

 private:
  uint64_t _cycle;
};

class TileStonneTraceMemoryNode : public TileNode {
 public:
  TileStonneTraceMemoryNode(onnx::NodeProto& node) : TileNode(node) {
    for (auto attribute : node.attribute()) {
      if (attribute.name() == "torchsim_trace_address") {
        trace_address.assign(attribute.ints().begin(), attribute.ints().end());
      }
    }
  }
  std::vector<uint64_t>& get_address() { return trace_address; }
  void print_node();

 private:
  std::vector<uint64_t> trace_address;
};
class TileStonneTraceLoadNode : public TileStonneTraceMemoryNode {
 public:
  using TileStonneTraceMemoryNode::TileStonneTraceMemoryNode;
};

class TileStonneTraceStoreNode : public TileStonneTraceMemoryNode {
 public:
  using TileStonneTraceMemoryNode::TileStonneTraceMemoryNode;
};