import functools
import itertools
import textwrap
import re
import math
import sympy
from collections import OrderedDict

from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import Kernel, KernelTemplate, ChoiceCaller, OpOverrides, CSE
from torch._inductor.ir import Buffer, IRNode, TemplateBuffer
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V, NullHandler, _ops as ops
from torch._inductor.utils import IndentedBuffer

from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel
from PyTorchSimFrontend.mlir.mlir_scheduling import SchedulerNode

from . import mlir_common

class MLIRTemplateKernel(MLIRKernel, BaseMLIRHardwareInfo):
    def __init__(self,
                 kernel_name,
                 input_nodes,
                 call_size,
                 kernel_group = None,
                 outer_func_name=None,
                 outer_func_render=None,
                 kernel_arg_attributes=None) -> None:
        super().__init__(kernel_group if kernel_group is not None else mlir_common.MLIRWrapperKenrelGroup())
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.call_size = call_size
        self.named_nodes = {}
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.outer_func_name = outer_func_name
        self.outer_func_render = outer_func_render
        self.kernel_arg_attributes = kernel_arg_attributes
        self.render_hooks = OrderedDict()
        self.buffer_names = dict()
        self.render_options = dict()
        self.tile_size = []
        self.loop_size = None
        self.is_template_kernel = True
        self.map_cse = CSE("#", self.suffix, name_prefix="template_map")
        self.const_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="template_const")
        self.alloc_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="template_alloc")

        # Overwrite ops
        self.load = self.load_epilogue
        self.store = self.store_epilogue

    def add_loop_info(self, mat_size, tile_size):
        for idx, (loop_size, stride) in enumerate(zip(mat_size, tile_size)):
            self.loop_info[f"index{idx}"] = [0, loop_size, stride]

    def gemmini_gemm_mapping(self, M, N, K):
        spad_size = self.spad_info["spad_size"] * self.vector_lane
        num_cores = self.num_cores
        precision = self.precision
        dim_I, dim_J, dim_K = M, N, K
        dim = self.vector_lane

        # split spad into 3/4 for input and 1/4 for output (only for mapping)
        # TODO: 3/4 and 1/4 are arbitrary numbers. We should find a better way to split the spad (auto-tune?)
        max_spad_rows = (spad_size * 3 // 4) // (dim * precision * 2) # 4 bytes per element, double buffer
        max_acc_rows = (spad_size // 4) // (dim * 4 * 2) # 4 bytes per element, double buffer

        dim_I_padded = (dim_I // dim + (dim_I % dim != 0)) * dim
        dim_J_padded = (dim_J // dim + (dim_J % dim != 0)) * dim
        dim_K_padded = (dim_K // dim + (dim_K % dim != 0)) * dim

        db_partitions_rows = max_spad_rows // 2
        db_mats_in_partition = db_partitions_rows // dim
        db_mats_in_acc = max_acc_rows // dim
        db_max_tile_i_j = int(math.sqrt(db_mats_in_acc))
        db_max_tile_k = db_mats_in_partition // db_max_tile_i_j

        tile_I = min(dim_I_padded // dim, math.ceil(dim_I / (db_max_tile_i_j * dim)))
        tile_J = min(dim_J_padded // dim, math.ceil(dim_J / (db_max_tile_i_j * dim)))
        tile_K = min(dim_K_padded // dim, math.ceil(dim_K / (db_max_tile_k * dim)))

        num_tiles = tile_I * tile_J
        if num_tiles < num_cores:
            increase_tile = math.ceil(num_cores / num_tiles)
            if dim_J > dim_I and dim_J > num_cores:
                tile_J *= increase_tile
            elif dim_I > dim_J and dim_I > num_cores:
                tile_I *= increase_tile
            num_tiles = tile_I * tile_J
        if num_tiles % num_cores != 0:
            increase_tile = num_tiles % num_cores
            if dim_J > dim_I and dim_J > num_cores:
                tile_J += increase_tile
            elif dim_I > dim_J and dim_I > num_cores:
                tile_I += increase_tile

        inner_I = math.ceil(dim_I_padded / tile_I)
        inner_J = math.ceil(dim_J_padded / tile_J)
        inner_K = math.ceil(dim_K_padded / tile_K)

        inner_I -= inner_I & (dim) - 1
        inner_J -= inner_J & (dim) - 1
        inner_K -= inner_K & (dim) - 1

        tile_I = math.ceil(dim_I / inner_I)
        tile_J = math.ceil(dim_J / inner_J)
        tile_K = math.ceil(dim_K / inner_K)

        return inner_I, inner_J, inner_K

    def gemm_combination_mapping(self, M, N, K, n_extra_node=0, pad_k=True):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer
        m_pad_factor = self.vector_lane if M > self.vector_lane else 8
        n_pad_factor = self.vector_lane if N > self.vector_lane else 8
        k_pad_factor = self.vector_lane if K > self.vector_lane else (8 if pad_k else 1)
        K = max(K, 8)
        M_padded = ((M + m_pad_factor - 1) // m_pad_factor) * m_pad_factor
        N_padded = ((N + n_pad_factor - 1) // n_pad_factor) * n_pad_factor
        K_padded = ((K + k_pad_factor - 1) // k_pad_factor) * k_pad_factor
        indexI, indexJ, indexK = (M_padded // self.vector_lane, N_padded // self.vector_lane, K_padded // self.vector_lane)

        max_used_spad_size = 0
        mapping = (self.vector_lane, self.vector_lane, self.vector_lane)
        tile_M_range = sympy.divisors(indexI) if M > self.vector_lane else [1]
        tile_N_range = sympy.divisors(indexJ) if N > self.vector_lane else [1]
        tile_K_range = sympy.divisors(indexK) if K > self.vector_lane else [1]
        maximize_i_j = 1 # reuse weight
        for k in tile_K_range:
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for j in tile_N_range:
                tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                for i in tile_M_range:
                    tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                    used_spad_size = (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M, tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane and maximize_i_j <= tile_M * tile_N:
                        max_used_spad_size = used_spad_size
                        maximize_i_j = tile_M * tile_N
                        mapping = (tile_M, tile_N, tile_K)
        return mapping

    def conv_combination_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node, pad_k=False)
        max_k_h_w = 1 # maximize kernel size
        for o_h in sympy.divisors(O_H):
            for o_w in sympy.divisors(O_W):
                for k_h in sympy.divisors(K_H):
                    for k_w in sympy.divisors(K_W):
                        i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                        i_w = 1 + (o_w - 1) * stride[1] + (k_w - 1) * dilation[1]
                        weight_size = k_w * k_h * K * N
                        input_size = i_w * i_h * M * K
                        output_size = o_w * o_h * M * N
                        used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                        weight_size_per_lane = self.get_spad_size_per_lane(k_w * k_h * K, N)
                        input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * M, K)
                        output_size_per_lane = self.get_spad_size_per_lane(o_w * o_h * M  * (1 + n_extra_node), N)
                        used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                        if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane and max_k_h_w <= k_h * k_w:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h * k_w
                            mapping = (k_h, k_w, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        return mapping

    def conv_multi_tile_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K * K_W, n_extra_node=n_extra_node, pad_k=False)
        max_k_h_w = K_W
        for o_h in sympy.divisors(O_H):
            for o_w in sympy.divisors(O_W):
                for k_h in sympy.divisors(K_H):
                    i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                    i_w = 1 + (o_w - 1) * stride[1] + (K_W - 1) * dilation[1]
                    weight_size = 1 * k_h * K * N
                    input_size = i_w * i_h * M * K
                    output_size = o_w * o_h * M * N
                    used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(1 * k_h * K, N)
                    input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * M, K)
                    output_size_per_lane = self.get_spad_size_per_lane(o_w * o_h * M  * (1 + n_extra_node), N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane and max_k_h_w <= k_h:
                        max_used_spad_size = used_spad_size
                        max_k_h_w = k_h
                        mapping = (k_h, K_W, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        return mapping

    def conv_single_batch_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(O_W, N, K, n_extra_node=n_extra_node, pad_k=False)
        max_k_h_w = 1
        for o_h in sympy.divisors(O_H):
            for k_h in sympy.divisors(K_H):
                for k_w in sympy.divisors(K_W):
                    i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                    i_w = 1 + (M - 1) * stride[1] + (k_w - 1) * dilation[1]
                    weight_size = k_w * k_h * K * N
                    input_size = i_w * i_h * k_w * K
                    output_size = M * o_h * N
                    used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(k_w * k_h * K, N)
                    input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * k_w, K)
                    output_size_per_lane = self.get_spad_size_per_lane(M * o_h  * (1 + n_extra_node), N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane and max_k_h_w <= k_h * k_w:
                        max_used_spad_size = used_spad_size
                        max_k_h_w = k_h * k_w
                        mapping = (k_h, k_w, o_h, M, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        return mapping

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        kernel_arg_attributes = self.kernel_arg_attributes
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        if kernel_arg_attributes is not None:
            for name, attr in kernel_arg_attributes:
                for idx in range(len(arg_attributes)):
                    if arg_attributes[idx][0] == name:
                        arg_attributes[idx][1] = attr
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom PyTorchSimFrontend.extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
        # generate the code to call this
        wrapper.generate_kernel_call(
            kernel_name if self.outer_func_name is None else self.outer_func_name,
            call_args, cuda=False)

    def codegen_body(self):
        def template_store():
            sram_var = self.store_info["sram_var"]
            dram_var = self.store_info["dram_var"]
            index_var = self.store_info["index_var"]
            tag_var = self.store_info["tag_var"]
            vlane_split_axis = self.store_info["vlane_split_axis"]
            vlane_stride = self.store_info["vlane_stride"]
            mlir_dtype = self.store_info["mlir_dtype"]
            dram_shape = self.store_info["dram_shape"]
            tile_shape = f"memref<{'x'.join(str(s) for s in self.store_info['tile_size'])}x{mlir_dtype}, 1>"
            zero_cse = self.get_const_cse(0)
            sram_index_var = ",".join([f"%{zero_cse}"] * self.store_info["tile_nr_dim"])
            tile_stride = self.store_info['tile_stride']
            code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 tag_var, dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_stores, code, assignment = False)
        self.body.splice(self.spad_buffer)
        self.body.splice(self.applys)
        self.body.splice(self.dma_loads)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        if len(self.stores._lines) == 0:
            template_store()
        self.body.splice(self.stores)
        self.body.splice(self.dma_stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.input_buffers[node.get_name()] = name

        extra_node = {}
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.output_buffers[node.get_name()] = name
                self.store_buffer_names.add(node.get_name())    #TODO: Is this enough not calling store() in mlir_common.py?
                if isinstance(node, SchedulerNode):
                    extra_node[node.get_name()] = node.node
                else:
                    extra_node[node.get_name()] = node
                self.buffer_names[node.get_name()] = self.store_info['sram_var']

        def hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=extra_node)
            return f"({', '.join(arg_defs)})"

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    # This function is a temporal function for convolution because currently convolution kernel is not considering padding.
    # Padding is done by python wrapper so the padded input size is manually applied here.
    def def_conv_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        padded_input_size: List[int] = [],
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.input_buffers[node.get_name()] = name

        self.extra_node = {}
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.output_buffers[node.get_name()] = name
                self.store_buffer_names.add(node.get_name())    #TODO: Is this enough not calling store() in mlir_common.py?
                self.extra_node[node.get_name()] = node
                self.buffer_names[node.get_name()] = self.store_info['sram_var']   #TODO: Buffer name fixed

        def kernel_hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=self.extra_node)
            arg_defs[0] = re.sub(r'(\d+)(?=xf32)', str(padded_input_size), arg_defs[0])
            return f"({', '.join(arg_defs)})"

        assert "<DEF_CONV_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_CONV_KERNEL>"] = kernel_hook
        return "<DEF_CONV_KERNEL>"

    # This function is for convolution wrapper function finalizing.
    def def_wrapper(self, only_store_buffer: bool = False, epilogue_buffer: str = False):
        def wrapper_hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=self.extra_node)
            wrapper_arg_defs = [arg.split('%')[1].split(':')[0] for arg in arg_defs]
            return f"({', '.join(wrapper_arg_defs)})"

        if "<DEF_CONV_WRAPPER>" not in self.render_hooks:
            self.render_hooks["<DEF_CONV_WRAPPER>"] = wrapper_hook
        return "<DEF_CONV_WRAPPER>"

    def get_conv_inputs(self):
        return self.kernel_group.args.input_buffers

    def get_conv_outputs(self):
        return {k: v for k, v in self.kernel_group.args.output_buffers.items() if v != 'REMOVED'}

    def output_name(self):
        # Cannot know the output name from the template, so we need to hook it
        def hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs()
            output = arg_defs[3]    #FIXME: Constant index used
            pattern = r"%(\w+):"
            output = re.search(pattern, output).group(1)
            return output
        assert "<OUPUT>" not in self.render_hooks
        self.render_hooks["<OUPUT>"] = hook
        return "<OUPUT>"

    def store_output(self, indent_size: int = 0):
        def hook():
            self.codegen_body()
            return textwrap.indent(self.body.getvalue(), " "*indent_size).strip()

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        self.render_hooks.move_to_end("<STORE_OUTPUT>", last=False) # Force order to be triggered first
        return "<STORE_OUTPUT>"

    def def_function(self):
        _, call_args, _ = self.kernel_group.args.python_argdefs()
        if self.outer_func_render is not None:
            partial_code, function_name = self.outer_func_render(input_args=call_args)
            return PartialRender(
                partial_code,
                self.render_hooks,
            ), function_name
        else:
            return None, None

    def def_global_vars(self):
        key = "<GLOBAL_VARS>"
        def hook():
            return textwrap.indent(self.global_vars.getvalue(), "").strip()

        assert key not in self.render_hooks
        self.render_hooks[key] = hook
        return key

    def def_local_vars(self):
        key = "<LOCAL_VARS>"
        def hook():
            code = IndentedBuffer()
            code.tabwidth = 2
            code.splice("\n")
            with code.indent():
                code.splice(self.const_buffer)
                code.splice(self.alloc_buffer)
            return code.getvalue()

        assert key not in self.render_hooks
        self.render_hooks[key] = hook
        return key

    def render(self, template, kwargs, define_function=None):
        # self.render_hooks = {}
        code = template.render(**kwargs)
        if define_function is not None:
            define_function(self)
        return PartialRender(
            code,
            self.render_hooks,
        )

    def get_spad_size_per_lane(self, tile_m, tile_n):
        size = tile_m * ((tile_n + self.vector_lane - 1) // self.vector_lane)
        return max(size, 2) # vector load/store

    def load_epilogue(self, name: str, index: sympy.Expr):
        load_dim = []
        if not isinstance(V.graph, NullHandler) and name in V.graph.graph_inputs:
            load_dim = V.graph.graph_inputs[name].layout.size
        index_var = self.store_info['index_var'] if len(load_dim) != 1 else 'tile_n'
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.input(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis if len(load_dim) != 1 else 0    # FIXME: Fixed split axis for 1d load dim
        vlane_stride = self.kernel_group.tile_desc.vlane_stride if len(load_dim) != 1 else 1    # FIXME: Fixed stride for 1d load dim
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        vshape = f", {vshape}" if tile_numel_per_lane > 1 else ""
        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
            tile_stride = self.store_info['tile_stride']
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)
            self.buffer_names[name] = sram_var
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     f"{name}_tag", dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_loads, code, assignment = False)

        # Load vector from sram
        sram_var = self.buffer_names[name]
        operation = "affine.vector_load" if tile_numel_per_lane > 1 else "affine.load"
        zero_var = self.get_const_cse(0)
        tile_indices = ",".join([f"%{zero_var}"] * self.store_info["tile_nr_dim"])
        line = f"{operation} %{sram_var}[{tile_indices}] : memref<{'x'.join(str(s) for s in self.store_info['tile_size'])}x{mlir_dtype}, 1>{shape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [tile_numel_per_lane, mlir_dtype])
        return out

    def store_epilogue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index_var = self.store_info['index_var']
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vlane_stride
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.store_info['tile_stride']
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        vshape = f", {vshape}" if tile_numel_per_lane > 1 else ""

        if name not in self.buffer_names:
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)
            self.buffer_names[name] = sram_var
        else:
            zero_cse = self.get_const_cse(0)
            sram_dims = len(tile_shape.split("x")) - 1
            sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)
        sram_var = self.buffer_names[name]

        operation = "affine.vector_store" if tile_numel_per_lane > 1 else "affine.store"
        zero_var = self.get_const_cse(0)

        _, operand_type = self.var_info[value]
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype, var_info=self.var_info)

        tile_indices = ",".join([f"%{zero_var}"] * self.store_info["tile_nr_dim"])
        line = f"{operation} %{value}, %{sram_var}[{tile_indices}] : {tile_shape}{vshape}"

        self.cse.generate(self.stores, line, assignment = False)
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.cse.generate(self.dma_stores, code, assignment = False)

    def get_scratchpad_buffer(self, dtype, name, tile_size_per_lane, dram_tile_shape, index_var, raw_index):
        return super().get_scratchpad_buffer(dtype, name, tile_size_per_lane, dram_tile_shape, index_var, raw_index, True)

class MLIRTemplateCaller(CUDATemplateCaller):
    def __str__(self):
        return f"MLIRTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"mlir_template_kernels.{self.name}"

class MLIRTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name, input_nodes, layout, input_reorder = None):
        """
        Baseclass for MLIR Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = [node for node in input_nodes if node is not None]
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"mlir_{self.name}"
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            kernel  = MLIRTemplateKernel(kernel_name=kernel_name, input_nodes=self.input_nodes, call_size=self.layout.size, kernel_group=None,
                                         outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                                         outer_func_render=self.outer_func_render if hasattr(self, 'outer_func_render') else None,
                                         kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None)
            code = self.render(kernel=kernel, **kwargs)

        kernel_hash_name = f"mlir_{self.name}_{next(self.index_counter)}"
        extra_args = []
        # create the BenchmarkRequest
        bmreq = MLIRBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: TemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
            kernel_name: str = kernel_hash_name,
            kernel_group: Optional[mlir_common.MLIRWrapperKenrelGroup] = None
        ):
            kernel = MLIRTemplateKernel(
                kernel_name=kernel_name,
                input_nodes=self.input_nodes,
                call_size=self.layout.size,
                kernel_group=kernel_group,
                outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                outer_func_render=functools.partial(
                    self.outer_func_render,
                    kernel_name=kernel_name
                ) if hasattr(self, 'outer_func_render') else None,
                kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None
            )

            kwargs = {
                'kernel': kernel,
                'template_buffer_node': template_node,
                'epilogue_nodes': epilogue_nodes
            }
            render = functools.partial(
                kernel.render,
                template=self,
                kwargs=kwargs
            )
            return kernel, render, self.codegen_header

        return MLIRTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def render(self, **kwargs) -> str:
        raise NotImplementedError