#pragma once
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "TileGraph.h"
#include "Instruction.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

enum class TileType{
  LOOP_INDEX_NODE,
  LOOP_END_NODE,
  LOAD_NODE,
  STORE_NODE,
  COMPUTE_NODE
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
  void print_node();
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

class TileComputeNode : public TileNode {
 public:
  TileComputeNode(onnx::NodeProto& node);
  uint32_t get_cycle() { return _cycle; }
  void print_node();

 private:
  std::map<std::string, std::shared_ptr<TileNode>> tile_map;
  uint32_t _cycle;
};

class TileMemoryNode : public TileNode {
 public:
  TileMemoryNode(onnx::NodeProto& node);
  addr_type get_base_addr() { return _base_addr; }
  std::vector<size_t> get_tile_size() { return _tile_size; }
  std::vector<size_t> get_tile_stride() { return _tile_stride; }
  void print_node();

 private:
  std::vector<size_t> _tile_size;
  std::vector<size_t> _tile_stride;
  std::vector<size_t> _stride_list;
  uint32_t _element_size;
  addr_type _base_addr = 0;
  std::string _base_addr_name;
};

class TileLoopNode : public TileNode {
 public:
  TileLoopNode(onnx::NodeProto& node);
  void add_body(std::shared_ptr<TileNode> body) { _body_node.push_back(body); }
  std::vector<std::shared_ptr<Tile>> get_tiles_from_iter(int iter);
  uint64_t get_start() { return _start; }
  uint64_t get_stride() { return _stride; }
  uint64_t get_end() { return _end; }
  void print_node();

 private:
  std::string _tile_index_name;
  uint64_t _stride;
  uint64_t _start;
  uint64_t _end;
  std::vector<std::shared_ptr<TileNode>> _body_node;
};

class TileLoopEndNode : public TileNode {
 public:
  TileLoopEndNode(onnx::NodeProto& node) : TileNode(node) {}
};

class TileGraphParser {
 public:
  TileGraphParser(std::string onnx_path);
  void initialize_tile(std::string op_type);
  std::shared_ptr<TileNode> get_top_loop();
  std::unique_ptr<TileGraph>& get_tile_graph() { return _tile_graph; }
 private:
  void register_tile(std::shared_ptr<TileNode> tile_node);
  void _tile_generate() {}
  void _base_addr_update() {}
  void _tile_index_generate() {}
  int _loop_stack_pointer = 0;
  std::map<std::string, std::shared_ptr<TileNode>> _output_map;
  std::vector<std::shared_ptr<TileNode>> _loop_nodes;
  std::vector<std::shared_ptr<TileNode>> _tile_vec;
  std::unique_ptr<TileGraph> _tile_graph;
};