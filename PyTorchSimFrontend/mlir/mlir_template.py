import functools
import itertools
import textwrap
import re
import os
import contextlib
import math
import sympy
from functools import reduce
import operator
from collections import OrderedDict

from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import KernelTemplate, CSE, DeferredLine
from torch._inductor.ir import Buffer, IRNode, TemplateBuffer, ChoiceCaller
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V, NullHandler, _ops as ops
from torch._inductor.utils import IndentedBuffer
from torch._inductor.codecache import write_atomic

import PyTorchSimFrontend.extension_codecache as extension_codecache
from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel, reduction_init, reduction_partial_combine_vec, is_welford_reduction
from PyTorchSimFrontend.mlir.mlir_scheduling import SchedulerNode
from torch._inductor.codegen import common

from PyTorchSimFrontend import extension_config
from . import mlir_common

# Configure logger for mlir_template module
logger = extension_config.setup_logger()

class IndentedBufferGroup:
    def __init__(self, kernel: 'MLIRTemplateKernel', prefix=""):
        self.kernel = kernel
        self.body = IndentedBuffer()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.applys = IndentedBuffer()
        self.dma_loads = IndentedBuffer()
        self.dma_stores = IndentedBuffer()
        self.spad_buffer = IndentedBuffer()
        self.cse = common.CSE("%", "", name_prefix=f"{prefix}")
        self.apply_cse = common.CSE("%", "", name_prefix=f"{prefix}apply")
        # Original buffers will be saved later in the 'with' block
        self.original_buffers = {}

    def set_buffers(self):
        self.kernel.loads = self.loads
        self.kernel.compute = self.compute
        self.kernel.stores = self.stores
        self.kernel.applys = self.applys
        self.kernel.dma_loads = self.dma_loads
        self.kernel.dma_stores = self.dma_stores
        self.kernel.spad_buffer = self.spad_buffer
        self.kernel.cse = self.cse
        self.kernel.apply_cse = self.apply_cse

    def restore_buffers(self):
        self.kernel.loads = self.original_buffers['loads']
        self.kernel.compute = self.original_buffers['compute']
        self.kernel.stores = self.original_buffers['stores']
        self.kernel.applys = self.original_buffers['applys']
        self.kernel.dma_loads = self.original_buffers['dma_loads']
        self.kernel.dma_stores = self.original_buffers['dma_stores']
        self.kernel.spad_buffer = self.original_buffers['spad_buffer']
        self.kernel.cse = self.original_buffers['cse']
        self.kernel.apply_cse = self.original_buffers['apply_cse']

    @contextlib.contextmanager
    def as_local(self):
        self.original_buffers = {
            'loads': self.kernel.loads,
            'compute': self.kernel.compute,
            'stores': self.kernel.stores,
            'applys': self.kernel.applys,
            'dma_loads': self.kernel.dma_loads,
            'dma_stores': self.kernel.dma_stores,
            'spad_buffer': self.kernel.spad_buffer,
            'cse': self.kernel.cse,
            'apply_cse': self.kernel.apply_cse,
        }
        try:
            self.set_buffers()
            with self.kernel.override_buffer_cse(buffer=self.compute, cse=self.cse):
                yield self
        finally:
            self.restore_buffers()

class MLIRTemplateKernel(MLIRKernel, BaseMLIRHardwareInfo):
    def __init__(self,
                 kernel_name,
                 input_nodes,
                 call_size,
                 kernel_group = None,
                 outer_func_name=None,
                 outer_func_render=None,
                 kernel_arg_attributes=None,
                 reason=None) -> None:
        super().__init__(kernel_group if kernel_group is not None else mlir_common.MLIRWrapperKenrelGroup())
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.call_size = call_size
        self.named_nodes = {}
        self.loop_info = {}
        self.outer_func_name = outer_func_name
        self.outer_func_render = outer_func_render
        self.kernel_arg_attributes = kernel_arg_attributes
        self.render_hooks = OrderedDict()
        self.buffer_names = dict()
        self.render_options = dict()
        self.tile_size = []
        self.loop_size = None
        self.map_cse = CSE("#", self.suffix, name_prefix="t_map")
        self.const_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="t_const")
        self.alloc_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="t_alloc")
        self.prologue_buffer_group = IndentedBufferGroup(self, prefix="prologue_")
        self.epilogue_buffer_group = IndentedBufferGroup(self, prefix="epilogue_")
        self.global_vars = IndentedBuffer()
        self.exception_nodes = {}
        # Reduction data structure
        self.reduction_epilogue_suffix = IndentedBuffer()
        self.reduction_fusion = False
        self.reduction_body_loop = None
        self.reduction_buffer_idx = 0
        self.reduction_info = {}
        self.reduction_epilogue_result = {}
        self.reduction_mean = []
        # Dim info
        self.dim_aliasing = {}
        self.reason = reason

    def reset(self, reason):
        self.__init__(
            self.kernel_name, self.input_nodes,
            self.call_size, self.kernel_group,
            self.outer_func_name, self.outer_func_render,
            self.kernel_arg_attributes, reason
        )

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

    def gemm_combination_mapping(self, M, N, K, n_extra_node=0, n_prologue_node=0, pad_k=True, min_tile=False, is_conv=False):
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer
        minimum_n_tile = self.num_cores if min_tile else 1
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
        for k in tile_K_range: # store tile candidates for manual mapping
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for i in tile_M_range:
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node) + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        dir_path = f"{extension_config.CONFIG_TORCHSIM_DIR}/validation/gemm_candidates"
                        os.makedirs(dir_path, exist_ok=True)
                        file_path = f"{dir_path}/gemm_{M}_{K}_{N}.txt"
                        line_to_write = f"{tile_M} {tile_K} {tile_N}\n"
                        try:
                            with open(file_path, "r") as f:
                                lines = f.readlines()
                        except FileNotFoundError:
                            lines = []
                        if line_to_write not in lines:
                            with open(file_path, "a") as f:
                                f.write(line_to_write)

        for k in tile_K_range: # heuristic search
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for i in tile_M_range:
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node) + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    n_tile = math.ceil(M / max(tile_M, 128)) * math.ceil(N / max(tile_N, 128))
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size and max_used_spad_size < used_spad_size and maximize_i_j <= tile_M * tile_N and n_tile >= minimum_n_tile and max(tile_N, 128) // max(tile_M, 128) < 10:
                        max_used_spad_size = used_spad_size
                        maximize_i_j = tile_M * tile_N
                        mapping = (tile_M, tile_N, tile_K)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (tile_M, tile_N, tile_K)))

        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_combination_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
        max_k_h_w = 1 # maximize kernel size
        max_o_h_w = 1 # maximize output size
        K = min(K, self.vector_lane)
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
                        check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                        if check_spad_size:
                            tile_candidates.append((used_spad_size, (k_h, k_w, o_h, o_w, M, N, K)))
                            if max_used_spad_size < used_spad_size and max_k_h_w <= k_h * k_w and max_o_h_w <= o_h * o_w:
                                max_used_spad_size = used_spad_size
                                max_k_h_w = k_h * k_w
                                max_o_h_w = o_h * o_w
                                mapping = (k_h, k_w, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")

        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_multi_tile_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K * K_W, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
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
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (k_h, K_W, o_h, o_w, M, N, K)))
                        if max_used_spad_size < used_spad_size and max_k_h_w <= k_h:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h
                            mapping = (k_h, K_W, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_single_batch_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(O_W, N, K, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
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
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (k_h, k_w, o_h, M, M, N, K)))
                        if max_used_spad_size < used_spad_size and max_k_h_w <= k_h * k_w:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h * k_w
                            mapping = (k_h, k_w, o_h, M, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def meta_kernel(self):
        kernel_arg_attributes = self.kernel_arg_attributes
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        if kernel_arg_attributes is not None:
            for name, attr in kernel_arg_attributes:
                for idx in range(len(arg_attributes)):
                    if arg_attributes[idx][0] == name:
                        arg_attributes[idx][1] = attr
        return arg_attributes

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
        # generate the code to call this
        wrapper.generate_kernel_call(
            kernel_name if self.outer_func_name is None else self.outer_func_name + f"_{len(call_args)}", call_args)

    def codegen_template_code(self, render, template_node, prologue_nodes, epilogue_nodes, tile_info):
        with self as kernel:
            _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()
            for node in [template_node, *prologue_nodes, *epilogue_nodes]:
                node.mark_run()

            # Partial codgen template nodes
            partial_code = render(kwargs={**render.keywords['kwargs'], 'tile_info': tile_info})

            # Swap load/store functions
            kernel.load = kernel.load_epilogue
            kernel.store = kernel.store_epilogue
            kernel.store_reduction = kernel.store_reduction_epilogue
            kernel.reduction = kernel.reduction_epilogue

            # Codegen prologue nodes
            if prologue_nodes:
                # Flush created varaibles, since template fusion doen't share variable
                with kernel.prologue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        [prologue_nodes[-1]], key=lambda x: int(x.is_reduction())
                    ).group
                    prologue_tile_desc = kernel.set_tile_size(kernel.prologue_info, prologue=True)
                    kernel.kernel_group.set_tile_info(prologue_tile_desc)
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group, list(self.dim_aliasing.values()))
                    for node in prologue_nodes:
                        # Reuse created spad
                        read_list = sorted([i.name for i in node.read_writes.reads])
                        candidate_found = False
                        # Why? There is a case that memdep.get_size() != data.get_size()
                        buf_dict = {}
                        buf_dict.update({val.name : val for val in V.graph.buffers})
                        buf_dict.update(V.graph.graph_inputs)
                        for candidate_read in read_list:
                            if candidate_read in buf_dict and reduce(operator.mul, buf_dict[candidate_read].get_size(), 1) == node.node.get_numel():
                                prologue_input_arg = candidate_read
                                candidate_found = True
                                break
                        assert(candidate_found)
                        assert(len(node.read_writes.writes)==1)
                        prologue_output_arg = list(node.read_writes.writes)[0].name
                        template_buf = self.kernel_group.args.input_buffers[prologue_output_arg]
                        target_buf = f"{template_buf}_buffer" # FIXME. How to pass spad buffer name?

                        # To skip the dma code gen
                        kernel.buffer_names[prologue_input_arg] = target_buf
                        kernel.buffer_names[prologue_output_arg] = target_buf

                        # Edge delete
                        kernel.kernel_group.args.input_buffers = {
                            (arg if buf != template_buf else prologue_input_arg): buf
                            for arg, buf in kernel.kernel_group.args.input_buffers.items()
                        }
                        node.codegen((vars, reduction_vars))

            # Codegen epilogue nodes
            tile_desc = kernel.set_tile_size(kernel.epilogue_info)
            kernel.kernel_group.set_tile_info(tile_desc)
            kernel.call_ranges = None
            if epilogue_nodes:
                with kernel.epilogue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        epilogue_nodes, key=lambda x: int(x.is_reduction())
                    ).group
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group, list(self.dim_aliasing.values()))
                    for node in epilogue_nodes:
                        node.codegen((vars, reduction_vars))

        with self as kernel:
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize_all()
            )

            # For consistency, white space could make wrong write_path
            buffer = IndentedBuffer()
            buffer.splice(src_code)
            src_code = buffer.getvalue()
            self._prepare_simulator_headers(src_code)
        meta_code = self.meta_kernel()
        return src_code, meta_code

    def make_choices(self, tile_candidates, render, template_node, prologue_nodes, epilogue_nodes):
        choices = []
        for tile_info in tile_candidates:
            # Compute Tile M, N, K DMA Tile M, N, K
            logger.debug(f"Auto-tune: Trying tile size: {list(tile_info)}")
            src_code, meta_code = self.codegen_template_code(render, template_node, prologue_nodes, epilogue_nodes, tile_info)
            bench_runner = self.run_bench([template_node], self.kernel_name, src_code)
            choices.append((bench_runner, src_code, meta_code, tile_info, self.loop_size))
            self.reset(reason=None)
        return choices

    def _log_autotune_result(self, best_choice, best_cycle):
        tile_size = best_choice[3]
        logger.debug(
            f"Auto-tune: Optimal tile size: {list(tile_size)}, "
            f"cycles: {best_cycle}"
        )

    def codegen_nodes(self, tile_candidates, render, template_node, prologue_nodes, epilogue_nodes):
        if "autotune" in extension_config.codegen_mapping_strategy and len(tile_candidates):
            src_code, meta_code, loop_size = self.autotune(tile_candidates, render, template_node, prologue_nodes, epilogue_nodes)
            self.loop_size = loop_size
        else:
            tile_info = tile_candidates[0] if tile_candidates else None
            src_code, meta_code = self.codegen_template_code(render, template_node, prologue_nodes, epilogue_nodes, tile_info)

        return src_code, meta_code

    def _prepare_simulator_headers(self, src_code):
        spad_end_symbol = f"int spad_end[0] __attribute__ ((section(\".spad\")));\n"
        spad_section_end_symbol = f"int spad_section_end[0] __attribute__ ((section(\".spad\"), aligned({self.spad_info['spad_size']*self.vector_lane})));"

        write_path = extension_codecache.get_write_path(src_code)
        if not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)
        spike_write_path = os.path.join(write_path, "global_var.h")
        gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
        if not os.path.exists(spike_write_path):
            write_atomic(spike_write_path, self.header.getvalue()+spad_end_symbol+spad_section_end_symbol)
        if not os.path.exists(gem5_write_path):
            write_atomic(gem5_write_path, self.gem5_header.getvalue())

    def codegen_prologue_body(self):
        body = IndentedBuffer()
        with self.prologue_buffer_group.as_local():
            body.splice(self.spad_buffer)
            body.splice(self.applys)
            body.splice(self.dma_loads)

            if (self.loads.getvalue() != '' or self.compute.getvalue() != '' or self.stores.getvalue() != ''):
                body.writelines(self.prologue_compute_body_loop.lines())
                compute_body = mlir_common.ParallelLoopBuffer()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                    compute_body.splice(self.stores)
                body.splice(compute_body)
            body.splice(self.dma_stores)
        return body

    def codegen_epilogue_body(self):
        def template_store():
            dram_var = self.epilogue_info["dram_var"]
            index_list = self.epilogue_info["dram_idx"]
            tile_desc = self.epilogue_info["dram_tile_desc"]
            code = self.def_dma_op("MVOUT", dram_var, index_list, tile_desc)
            self.cse.generate(self.dma_stores, code, assignment = False)

        body = IndentedBuffer()
        with self.epilogue_buffer_group.as_local():
            # Do dma store first to overlap epilogue nodes
            if self.reduction_fusion:
                if len(self.stores._lines) == 0:
                    template_store()
                    body.splice(self.dma_stores)
                    self.dma_stores.clear()
            body.splice(self.spad_buffer)
            body.splice(self.applys)
            body.splice(self.dma_loads)
            body.writelines(self.compute_body_loop.lines())
            compute_body = mlir_common.ParallelLoopBuffer()
            with contextlib.ExitStack() as stack:
                stack.enter_context(compute_body.indent(attribute="{inner_loop=false}",suffix=self.compute_body_loop.epilogue_line()))
                if self.reduction_fusion:
                    compute_body.splice(self.masks)
                    compute_body.writelines(self.reduction_body_loop.lines())
                    stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                else:
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                    if len(self.stores._lines) == 0:
                        template_store()
                compute_body.splice(self.stores)
            if (compute_body.getvalue()):
                body.splice(compute_body)
            body.splice(self.dma_stores)
            body.splice(self.reduction_epilogue_suffix)
        return body

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
                self.buffer_names[node.get_name()] = self.epilogue_info['sram_var']

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
                self.buffer_names[node.get_name()] = self.epilogue_info['sram_var']   #TODO: Buffer name fixed

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

    def load_input(self, indent_size: int = 0):
        def hook():
            code = IndentedBuffer()
            prologue_code = self.codegen_prologue_body()
            if prologue_code.getvalue():
                input_dma_code = self.def_dma_op("MVIN", self.prologue_info["input_dram_var"], self.prologue_info["input_idx"],
                                self.prologue_info["input_tile_desc"], subtile_size=self.prologue_info["input_subtile_size"], async_type=False)
                weight_dma_code = self.def_dma_op("MVIN", self.prologue_info["weight_dram_var"], self.prologue_info["weight_idx"],
                                self.prologue_info["weight_tile_desc"], subtile_size=self.prologue_info["weight_subtile_size"], async_type=False)
                if (self.prologue_info["is_input_fused"]):
                    code.splice(input_dma_code)
                    code.splice(prologue_code)
                    code.splice(weight_dma_code)
                else:
                    code.splice(weight_dma_code)
                    code.splice(prologue_code)
                    code.splice(input_dma_code)
            else:
                dma_code = self.def_dma_op("MVIN", self.prologue_info["input_dram_var"], self.prologue_info["input_idx"],
                                self.prologue_info["input_tile_desc"], subtile_size=self.prologue_info["input_subtile_size"], async_type=False)
                code.splice(dma_code)
                dma_code = self.def_dma_op("MVIN", self.prologue_info["weight_dram_var"], self.prologue_info["weight_idx"],
                                self.prologue_info["weight_tile_desc"], subtile_size=self.prologue_info["weight_subtile_size"], async_type=False)
                code.splice(dma_code)
            code = textwrap.indent(code.getvalue(), " "*indent_size).strip()
            return code

        assert "<PREPARE_INPUT>" not in self.render_hooks
        self.render_hooks["<PREPARE_INPUT>"] = hook
        self.render_hooks.move_to_end("<PREPARE_INPUT>", last=False) # Force order to be triggered first
        return "<PREPARE_INPUT>"

    def store_output(self, indent_size: int = 0):
        def hook():
            epilogue_code = self.codegen_epilogue_body()
            return textwrap.indent(epilogue_code.getvalue(), " "*indent_size).strip()

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        self.render_hooks.move_to_end("<STORE_OUTPUT>", last=False) # Force order to be triggered first
        return "<STORE_OUTPUT>"

    def reduction_output(self, indent_size: int = 0):
        def hook():
            return textwrap.indent(self.reductions_suffix.getvalue(), " "*indent_size).strip()

        assert "<REDUCTION_OUTPUT>" not in self.render_hooks
        self.render_hooks["<REDUCTION_OUTPUT>"] = hook
        return "<REDUCTION_OUTPUT>"

    def def_function(self):
        _, call_args, _, _ = self.kernel_group.args.python_argdefs()
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

    def def_local_vars(self, indent_size=0):
        key = "<LOCAL_VARS>"
        def hook():
            code = IndentedBuffer()
            code.tabwidth = 1
            code.splice(self.const_buffer)
            code.splice(self.alloc_buffer)
            return textwrap.indent(code.getvalue(), " "*indent_size).strip()

        assert key not in self.render_hooks
        self.render_hooks[key] = hook
        return key

    def def_dma_op(self, dma_type, dram_var:str, index_list:list, tile_desc:mlir_common.MLIRMultiDimTile,
                   subtile_size:list=[], async_type=None, indent_size=0):
        # Prepare code block
        local_code = IndentedBuffer()
        with self, self.override_buffer_cse(buffer=local_code, cse=self.apply_cse):
            index_var = self.parse_index_list(index_list, offset=tile_desc.offset)
            node_layout = self.named_nodes[dram_var].get_layout()
            if dram_var in self.exception_nodes:
                numel = self.exception_nodes[dram_var]["numel"]
            else:
                numel = self.get_arg_info(self.named_nodes[dram_var].get_name()).get_numel()
            mlir_dtype = mlir_common.DTYPE_TO_MLIR[node_layout.dtype]
            dram_shape = f"memref<{numel}x{mlir_dtype}>"
            dram_stride = []
            for idx in index_list:
                if idx.is_Mul:
                    dram_stride.append(int(idx.args[0]))
                elif idx == sympy.Symbol("c0"):
                    dram_stride.append(0)
                elif not idx.is_Number:
                    dram_stride.append(1)
                else:
                    dram_stride.append(0)

            sram_var = tile_desc.get_name()
            tile_shape = tile_desc.get_mlir_shape(mlir_dtype)
            tile_stride = tile_desc.get_tile_stride()
            vlane_split_axis = tile_desc.vmap.vlane_split_axis
            vlane_stride = tile_desc.vmap.vlane_stride

            zero_cse = self.get_const_cse(0, "index")
            sram_index_var = ", ".join([f"%{str(zero_cse)}"]*tile_desc.get_nr_dim())

            attribute_parts = [f"dram_stride={dram_stride}", f"sram_stride={tile_stride}", "padding=0"]
            if subtile_size:
                attribute_parts.append(f"subtile_size={subtile_size}, async={int(async_type) if async_type is not None else 1}")
            attribute = "  {" + ", ".join(attribute_parts) + "}"
            code = self.get_dma_code(dma_type, vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                    dram_shape, tile_shape, "")
            local_code.writeline(code)
            local_code.writeline(attribute)
        return textwrap.indent(local_code.getvalue(), " "*indent_size).strip()

    def def_sram_buffer(self, dram_name, tile_desc, id=0, indent_size=0):
        # Prepare code block
        with self:
            dtype = self.named_nodes[dram_name].get_layout().dtype
            tile_shape = tile_desc.get_mlir_shape(mlir_common.DTYPE_TO_MLIR[dtype])
            buffer_name = self.allocate_sram_buffer(dtype, dram_name, tile_desc, id, forced_name=dram_name)
            code = f"%{tile_desc.name} = memref.get_global @{buffer_name} : {tile_shape}"
        return textwrap.indent(code, " "*indent_size).strip()

    def render(self, template, kwargs, define_function=None):
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
        dram_var = self.kernel_group.args.input(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        # Want to use tile_desc from epilogue_info
        with self.override_buffer_cse(buffer=self.applys, cse=self.apply_cse):
            index_var = self.parse_indices(index)
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()]
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.kernel_group.tile_desc.get_tile_stride()

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, self.kernel_group.tile_desc, index)
            attribute = f"{{dram_stride={dram_stride}, sram_stride={tile_stride}, padding=0}}"
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     dram_shape, tile_shape, attribute)
            self.cse.generate(self.dma_loads, code, assignment = False)
            self.buffer_names[name] = sram_var
        else:
            sram_var = self.buffer_names[name]

        # Load vector from sram
        zero_var = self.get_const_cse(0)
        if not self.reduction_fusion:
            compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
            with self.override_buffer_cse(buffer=self.loads):
                out = ops._load(compute_vec_size, mlir_dtype, sram_var, compute_index_var, tile_shape)
        else: # For reduction case
            reduce_size = self.reduction_nr_outer_loop
            vsize = compute_vec_size//reduce_size

            if compute_vec_size > 1:
                with self.override_buffer_cse(buffer=self.global_vars, cse=self.map_cse):
                    map_var = ops.affine_map(["d0", "d1"], f"d0 + d1*{(self.r_tile_size)}")
                with self.override_buffer_cse(buffer=self.loads):
                    offset = ops.affine_apply(map_var, [self.compute_idx, self.reduction_loop_idx])
                compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{offset}"])

            with self.override_buffer_cse(buffer=self.loads):
                out = ops._load(vsize, mlir_dtype, sram_var, compute_index_var, tile_shape)
            self.register_var_info(out, [self.compute_body_loop.step, mlir_dtype])
        return out

    def store_epilogue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        dram_var = self.kernel_group.args.output(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        with self.override_buffer_cse(buffer=self.applys, cse=self.apply_cse):
            index_var = self.parse_indices(index)
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()]
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.kernel_group.tile_desc.get_tile_stride()

        if name not in self.buffer_names:
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, self.kernel_group.tile_desc, index)
            self.buffer_names[name] = sram_var
            store_force = False
        else:
            zero_cse = self.get_const_cse(0)
            sram_dims = len(tile_shape.split("x")) - 1
            sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)
            store_force = True
        sram_var = self.buffer_names[name]
        zero_var = self.get_const_cse(0)

        _, operand_type = self.var_info[value]
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype)
        compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
        # Generate vector load instruction
        buffer_name = name if not store_force else None
        with self.override_buffer_cse(buffer=self.stores):
            ops._store(value, sram_var, compute_index_var, tile_shape, buffer_name=buffer_name)

        # Generate DMA instruction
        attribute = f"{{dram_stride={dram_stride}, sram_stride={tile_stride}, padding=0}}"
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 dram_shape, tile_shape, attribute)
        self.dma_stores.writeline(DeferredLine(name, code))

    def reduction_epilogue(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        if is_welford_reduction(reduction_type):
            if reduction_type == "welford_combine":
                raise NotImplementedError("welford_combine")
            else:
                assert reduction_type == "welford_reduce"
                type_name = mlir_common.DTYPE_TO_MLIR[dtype]
                reduction_key = src_dtype, reduction_type, value
                sum = self.reduction_epilogue(dtype, src_dtype, "sum", value)
                sqr_sum = self.reduction_epilogue(dtype, src_dtype, "sum", ops.mul(value, value))
                self.welford_reduce_out = (sum, sqr_sum, None)
                return sum, sqr_sum, None

        # Check duplicated reductions
        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_epilogue_result:
            return self.reduction_epilogue_result[reduction_key]

        # Reduction fusion codegen part
        vec_size = self.compute_body_loop.step
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        new_tile_size = self.kernel_group.tile_desc.get_tile_size()[:-1] + [vec_size]
        new_vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        new_vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        local_tile_desc = mlir_common.MLIRMultiDimTile(new_tile_size, self.vector_lane, new_vlane_split_axis, new_vlane_stride, vec_size)

        tile_shape = local_tile_desc.get_mlir_shape(type_name)
        vshape = local_tile_desc.get_mlir_vshape(type_name)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        name = f"{reduction_type}_buffer{self.reduction_buffer_idx}"
        self.reduction_buffer_idx += 1
        index = "dummy_index" # Not used
        sram_var, _ = self.get_scratchpad_buffer(dtype, name, local_tile_desc, index, self.const_buffer)
        self.reduction_epilogue_result[reduction_key] = sram_var

        # Load partial result
        zero_var_list = [f"%{self.get_const_cse(0)}"] * local_tile_desc.get_nr_dim()
        zero_var_list[-2] = f"%{self.reduction_loop_idx}"
        compute_index_var = ", ".join(zero_var_list)
        with self.override_buffer_cse(buffer=self.loads):
            out = ops._load(vec_size, type_name, sram_var, compute_index_var, tile_shape)
        # Reduction body codegen
        with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
            init = ops.constant(reduction_init(reduction_type, dtype), type_name)
            init_vec = ops.broadcast(init, compute_vec_size)
            init_vec2 = ops.broadcast(init, local_tile_desc.get_numel_per_lane())
            ops._store(init_vec2, sram_var, ", ".join([f"%{self.get_const_cse(0)}"] * local_tile_desc.get_nr_dim()), tile_shape)

        mask_shape, mask_var = self.get_mask()
        if mask_var is not None:
            value = ops.where(mask_var, value, init_vec)

        result = reduction_partial_combine_vec(reduction_type, value, out)

        # Store partial result
        ops._store(result, sram_var, compute_index_var, tile_shape) # Need to be placed after partial reduction
        self.reduction_info[sram_var] = [reduction_type, local_tile_desc]
        return sram_var

    def store_reduction_epilogue(self, name, index, value):
        dram_var = self.kernel_group.args.output(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        with self.override_buffer_cse(buffer=self.reductions_suffix, cse=self.apply_cse):
            index_var = self.parse_indices(index, comments="// Store reduction")
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()][:-1] # Assume that there is only one reduction axis
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride

        # Create final buffer descriptor
        nr_outer_loop = self.reduction_nr_outer_loop
        tile_size = self.kernel_group.tile_desc.get_tile_size()[:-1]
        final_tile_desc = mlir_common.MLIRMultiDimTile(tile_size, self.vector_lane, vlane_split_axis, vlane_stride*nr_outer_loop*2)
        final_tile_shape = final_tile_desc.get_mlir_shape(mlir_dtype)
        final_tile_stride = final_tile_desc.get_tile_stride()
        sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, final_tile_desc, index, buffer=self.const_buffer)

        # Set partial buffer descriptor
        partial_tile_desc = self.reduction_info[value][1]
        partial_vec_size = partial_tile_desc.get_compute_vec_size()
        partial_vshape = partial_tile_desc.get_mlir_vshape(mlir_dtype)
        partial_tile_shape = partial_tile_desc.get_mlir_shape(mlir_dtype)

        # Prepare constant
        with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
            init = ops.constant(reduction_init(self.reduction_info[value][0], dtype), mlir_dtype)
            init_vec = ops.broadcast(init, partial_vec_size)
            init_vec2 = ops.broadcast(init, 2)

        partial_zero_var_list = [f"%{self.get_const_cse(0)}"] * partial_tile_desc.get_nr_dim()
        final_zero_var_list = [f"%{self.get_const_cse(0)}"] * final_tile_desc.get_nr_dim()
        for i in range(self.reduction_body_loop.size):
            # Load partial result
            with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                body_index_var = ops.constant(i, "index")
                partial_zero_var_list[-2] = f"%{body_index_var}"
                compute_index_var = ",".join(partial_zero_var_list)

            with self.override_buffer_cse(buffer=self.reductions_suffix):
                out = ops._load(partial_vec_size, mlir_dtype, value, compute_index_var, partial_tile_shape)
                ops._store(init_vec, value, compute_index_var, partial_tile_shape) # Clear the partial buffer to zero

                # 2 step reduction
                new_vec_size = 2
                new_reduced_shape = f"vector<{new_vec_size}x{mlir_dtype}>"
                reduction_type = self.reduction_info[value][0]
                out = ops.multi_reduction(out, init_vec2, partial_vec_size, new_vec_size, partial_vshape, reduction_type, mlir_dtype)

            out2 = self.cse.generate(self.reductions_suffix, f"vector.shuffle %{out}, %{out} [1, 0] : {new_reduced_shape}, {new_reduced_shape}")
            self.register_var_info(out2, [new_vec_size, mlir_dtype])

            with self.override_buffer_cse(buffer=self.reductions_suffix):
                out = reduction_partial_combine_vec(self.reduction_info[value][0], out, out2)

                if self.welford_reduce_out is not None:
                    # NOTE: It not a real welford algorithm... We just used E(X^2) - E(X)^2
                    divider = ops.constant(float(self.r_dim_size), "f32")
                    if self.buffer_types[name][1] > 1:
                        divider_vec = ops.broadcast(divider, new_vec_size)
                    else:
                        divider_vec = divider

                    if self.current_node.node.origin_node: # FIXME: This is a temporary solution
                        # mean = SUM(X) / N
                        self.reduction_mean.append(ops.truediv(out, divider_vec))
                        out = self.reduction_mean[i]
                    else:
                        # m2 = (E(X^2) - E(X)^2) * N
                        sqr_mean = ops.truediv(out, divider_vec)
                        mean_sqr = ops.mul(self.reduction_mean[i], self.reduction_mean[i])
                        variance = ops.sub(sqr_mean, mean_sqr)
                        m2 = ops.mul(variance, divider_vec)
                        out = m2

                final_zero_var_list[-1] = f"%{body_index_var}"
                final_compute_index_var = ",".join(final_zero_var_list)
                ops._store(out, sram_var, final_compute_index_var, final_tile_shape, buffer_name=name)

        # MVOUT Encoding
        # Generate DMA instruction
        attribute = f"{{dram_stride={dram_stride}, sram_stride={final_tile_stride}, padding=0}}"
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                dram_shape, final_tile_shape, attribute)
        self.reductions_suffix.writeline(DeferredLine(name, code))

    def set_tile_size(self, template_fusion_info, prologue=False):
        tile_desc = template_fusion_info["dram_tile_desc"]
        if "dim_aliasing" in template_fusion_info:
            self.dim_aliasing = template_fusion_info["dim_aliasing"]

        if 'nr_rdim' in template_fusion_info and template_fusion_info['nr_rdim']==1:
            tile_desc.nr_rdim = 1
            numel_per_lane = tile_desc.get_numel_per_lane()
            r_tile_size = tile_desc.get_tile_size()[-1]
            nr_outer_loop = (numel_per_lane + r_tile_size-1) // r_tile_size
            tile_desc.vmap.forced_vec_size = nr_outer_loop * 32 # Why? Emprically selected, other option failed to functionality...

            self.reduction_fusion = True
            self.r_tile_size = tile_desc.get_tile_size()[-1]
            self.r_dim_size = template_fusion_info['r_dim_size']
            self.reduction_nr_outer_loop = nr_outer_loop
            self.reduction_loop_idx = self.register_var_cse("reduce_loop_idx", 1, "index")
            self.compute_body_loop.size = r_tile_size
            self.compute_body_loop.step = tile_desc.get_compute_vec_size() // nr_outer_loop
            self.reduction_body_loop = mlir_common.LoopLevel(self.reduction_loop_idx, nr_outer_loop)
        else:
            tile_desc.vmap.forced_vec_size = 64

            if prologue:
                self.prologue_compute_body_loop.size = tile_desc.get_numel_per_lane()
                self.prologue_compute_body_loop.step = tile_desc.get_compute_vec_size()
            else:
                self.compute_body_loop.size = tile_desc.get_numel_per_lane()
                self.compute_body_loop.step = tile_desc.get_compute_vec_size()
        return tile_desc

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
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
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
            prologue_nodes: Optional[List[IRNode]] = None,
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
                'epilogue_nodes': epilogue_nodes,
                'prologue_nodes': prologue_nodes,
            }
            render = functools.partial(
                kernel.render,
                template=self,
                kwargs=kwargs
            )
            tile_candidates = self.get_tile_candidates(**kwargs)[:extension_config.codegen_autotune_template_topk]
            return kernel, tile_candidates, render

        return MLIRTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            False,  # supports_epilogue_fusion
            self,
            kwargs,
            "" # Currently Empty description
        )

    def get_tile_candidates(self, **kwargs):
        return []

    def render(self, **kwargs) -> str:
        raise NotImplementedError