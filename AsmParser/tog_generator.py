import os
import sys
import importlib.util
from pathlib import Path
from collections import defaultdict

if __name__ == "__main__":
    from onnx_utility import node, loop_index_node, loop_end_node, load_node, store_node, memory_wait_node, compute_node, connect_nodes, dump_onnx_graph
    from onnx_utility import stonne_node, stonne_trace_compute_node, stonne_trace_load_node, stonne_trace_store_node
else:
    from AsmParser.onnx_utility import node, loop_index_node, loop_end_node, load_node, store_node, memory_wait_node, compute_node, connect_nodes, dump_onnx_graph
    from AsmParser.onnx_utility import stonne_node, stonne_trace_compute_node, stonne_trace_load_node, stonne_trace_store_node


def import_module_from_path(module_name, path):
    module_path = Path(path)  # Convert to Path object for safety
    if not module_path.exists() or not module_path.is_file():
        raise FileNotFoundError(f"No such file: '{module_path}'")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load module from path: '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

class tog_generator:
    BaseNodeKind = 0
    ComputeNodeKind = 1
    LoopNodeKind = 2
    DMANodeKind = 3
    DMAWaitNodeKind = 4
    StonneNodeKind = 5
    StonneTraceCompute= 6
    StonneTraceLoad = 7
    StonneTraceStore = 8
    def __init__(self, origins={"Unknown"}) -> None:
        self.module_name = "tile_operation_graph"
        self.module = None
        self.raw_graph = {}
        self.node_depth_stack = [[]]
        self.node_depth_pointer = 0
        self.node_dict = {}
        self.parent_to_children = defaultdict(list)
        self.new_node_id = 0
        self.loop_end_stack = []
        self.origins = origins

    def append_depth_stack(self, node):
        self.node_depth_stack[self.node_depth_pointer].append(node)

    def increase_depth_stack(self):
        if (len(self.node_depth_stack) == self.node_depth_pointer + 1):
            self.node_depth_stack.append([])
        self.node_depth_pointer += 1

    def decrease_depth_stack(self):
        self.node_depth_pointer -= 1

    def load_file(self, path):
        self.module = import_module_from_path(self.module_name, path)
        if hasattr(self.module, "graph"):
            self.raw_graph = self.module.graph
            self.parse_graph()

    def _create_node(self, dump_data):
        node_id = dump_data["node_id"]
        node_type = dump_data["node_type"]

        new_node = None
        new_end_node = None
        if node_type == self.BaseNodeKind:
            new_node = node(node_id)
        elif node_type == self.ComputeNodeKind:
            cycle = dump_data["compute_cycle"]
            compute_type = dump_data["compute_type"]
            new_node = compute_node(cycle=cycle, compute_type=compute_type, node_id=node_id)
        elif node_type == self.LoopNodeKind:
            loop_start = dump_data["loop_start"]
            loop_end = dump_data["loop_end"]
            loop_step  = dump_data["loop_step"]
            loop_idx = dump_data["loop_index"]
            loop_type = dump_data["loop_type"]
            new_node = loop_index_node(loop_idx, [loop_start, loop_end, loop_step, loop_type], node_id)
            new_end_node = loop_end_node(loop_idx, self.new_node_id)
            new_end_node.parent = dump_data["parents"][0]
            self.new_node_id += 1
        elif node_type == self.DMANodeKind:
            tile_info = {}
            tile_info["base_addr"] = dump_data["base_address"]
            tile_info["tile_size"] = dump_data["tile_size"]
            tile_info["tile_stride"] = dump_data["tile_stride"]
            tile_info["element_size"] = dump_data["element_size"]
            tile_info["tag_idx_list"] = dump_data["tag_idx_list"]
            tile_info["tag_stride_list"] = dump_data["tag_stride_list"]
            tile_info["loop_idx_list"] = dump_data["loop_idx_list"]
            tile_info["loop_stride_list"] = dump_data["loop_stride_list"]
            tile_info["is_async"] = dump_data["is_async"]
            tile_info["indirect_mode"] = dump_data["indirect_mode"]
            is_write = dump_data["is_write"]
            if is_write:
                new_node = store_node(tile_info, node_id=node_id)
            else:
                new_node = load_node(tile_info, node_id=node_id)
        elif node_type == self.DMAWaitNodeKind:
            tile_info = {}
            tile_info["tag_idx_list"] = dump_data["tag_idx_list"]
            tile_info["tag_stride_list"] = dump_data["tag_stride_list"]
            tile_info["tag_divider_list"] = dump_data["tag_divider_list"]
            tile_info["base_addr"] = dump_data["base_address"]
            new_node = memory_wait_node(tile_info, node_id=node_id)
        elif node_type == self.StonneNodeKind:
            new_node = stonne_node(dump_data, node_id=node_id)
        elif node_type == self.StonneTraceCompute:
            new_node = stonne_trace_compute_node(dump_data['trace_compute_cycle'], node_id=node_id)
        elif node_type == self.StonneTraceLoad:
            new_node = stonne_trace_load_node(dump_data['trace_address'], node_id=node_id)
        elif node_type == self.StonneTraceStore:
            new_node = stonne_trace_store_node(dump_data['trace_address'], node_id=node_id)
        else:
            print("Unexpected node_type :", node_type)
            exit(1)

        # add new meta data
        if node_id == 0:
            new_node.parent = -1
        else:
            new_node.parent = dump_data["parents"][0]

        return new_node, new_end_node

    def create_node(self, dump_data, prev_node):
        node_id = dump_data["node_id"]
        node_type = dump_data["node_type"]
        parent_node = None
        # add new meta data
        if node_id == 0:
            parent_node = -1
        else:
            parent_node = dump_data["parents"][0]

        new_node, new_end_node = self._create_node(dump_data)

        # Return
        if not prev_node:
            self.node_dict[new_node.id] = new_node
            return new_node

        if prev_node[-1].parent == new_node.parent:
            # Handle special cases
            if isinstance(prev_node[-1], load_node) and isinstance(new_node, load_node):
                connect_nodes(prev_node[-1].get_parent()[-1], new_node)
            elif isinstance(prev_node[-1], memory_wait_node) and isinstance(new_node, memory_wait_node):
                connect_nodes(prev_node[-1].get_parent()[-1], new_node)
            elif isinstance(prev_node[-1], load_node) and isinstance(new_node, compute_node) or \
                 isinstance(prev_node[-1], memory_wait_node) and isinstance(new_node, compute_node):
                for pn in prev_node:
                    if isinstance(pn, load_node) or isinstance(pn, memory_wait_node):
                        connect_nodes(pn, new_node)
            else:
                connect_nodes(prev_node[-1], new_node)
        elif prev_node[-1].id == new_node.parent:
            connect_nodes(prev_node[-1], new_node)
        else:
            last_end_node = self.loop_end_stack.pop()
            self.decrease_depth_stack()
            for current_depth_node in prev_node[::-1]:
                connect_nodes(current_depth_node, last_end_node)
                if isinstance(current_depth_node, load_node):
                    continue
                break
            while True:
                self.node_dict[last_end_node.id] = last_end_node
                if last_end_node.parent == new_node.parent:
                    connect_nodes(last_end_node, new_node)
                    break
                end_node = self.loop_end_stack.pop()
                self.decrease_depth_stack()
                connect_nodes(last_end_node, end_node)
                last_end_node = end_node
        if (node_type == self.LoopNodeKind):
            self.loop_end_stack.append(new_end_node)
            # Increase loop depth
            self.increase_depth_stack()
        self.append_depth_stack(new_node)
        self.node_dict[new_node.id] = new_node
        return new_node

    def parse_graph(self):
        # Create nodes
        prev_node = []
        self.new_node_id = len(self.raw_graph.values()) + 1
        for value in self.raw_graph.values():
            new_node = self.create_node(value, prev_node)
            if not prev_node or prev_node[-1].parent == new_node.parent:
                prev_node.append(new_node)
            else:
                prev_node = [new_node]

        prev_node = prev_node[-1]
        # Link remain end node
        while self.loop_end_stack:
            end_node = self.loop_end_stack.pop()
            self.decrease_depth_stack()
            self.node_dict[end_node.id] = end_node
            connect_nodes(prev_node, end_node)
            prev_node = end_node

    def generate_tile_graph(self, name="tile_graph", cycle_list=list, x_offset=int, w_offset=int, vector_lane=int, stonneGraph=False):
        node_list = list(self.node_dict.values())[1:]
        if len(node_list):
            node_list[0].set_parent([])
            for iter_node in self.node_dict.values():
                if isinstance(iter_node, compute_node):
                    if cycle_list:
                        iter_node.torchsim_cycle = cycle_list.pop(0)
                    else:
                        print("[TOGGen] Error compute cycle timing is missing...!")
                        iter_node.torchsim_cycle = 10
                    # FIXME.
                    if iter_node.torchsim_compute_type > 0:
                        is_preload = iter_node.torchsim_compute_type == 2
                        offset = w_offset if is_preload else x_offset
                        iter_node.torchsim_overlapping_cycle = max(iter_node.torchsim_cycle - offset, 0)

        origin_info = self.origins if isinstance(self.origins, str) else "_".join(map(str, self.origins))
        onnx_node_list = [node.to_onnx() for node in node_list] # Exclude root node
        dump_onnx_graph(name, onnx_node_list, vector_lane, origin_info, stonneGraph=stonneGraph)

if __name__ == "__main__":
    t = tog_generator()
    t.load_file("/tmp/torchinductor/tmp/sz6qi7bqkxn/csz6qi7bqkxnam5sxok4l4sppddjkijq5rd55s4qvdutd5ni73fc_tog.py")
    t.generate_tile_graph("./tile_graph.onnx", cycle_list=[1,1,1,1,1], x_offset=0, w_offset=0, vector_lane=128)