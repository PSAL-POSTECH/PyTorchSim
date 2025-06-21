import functools
import itertools
import textwrap
import re
import contextlib
import math
import sympy
from collections import OrderedDict

from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import Kernel, KernelTemplate, ChoiceCaller, OpOverrides, CSE, DeferredLine
from torch._inductor.ir import Buffer, IRNode, TemplateBuffer, Pointwise
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V, NullHandler, _ops as ops
from torch._inductor.utils import IndentedBuffer

from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel, reduction_init, reduction_partial_combine_vec, reduction_combine_vec, is_welford_reduction
from PyTorchSimFrontend.mlir.mlir_scheduling import SchedulerNode

from . import mlir_common

class IndentedBufferGroup:
    def __init__(self, kernel: 'MLIRTemplateKernel'):
        self.kernel = kernel
        self.body = IndentedBuffer()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.applys = IndentedBuffer()
        self.dma_loads = IndentedBuffer()
        self.dma_stores = IndentedBuffer()
        self.spad_buffer = IndentedBuffer()

    def set_buffers(self):
        self.kernel.loads = self.loads
        self.kernel.compute = self.compute
        self.kernel.stores = self.stores
        self.kernel.dma_loads = self.dma_loads
        self.kernel.dma_stores = self.dma_stores
        self.kernel.spad_buffer = self.spad_buffer

    @contextlib.contextmanager
    def as_local(self):
        yield self

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
        self.prologue_buffer_group = IndentedBufferGroup(self)
        self.epilogue_buffer_group = IndentedBufferGroup(self)
        self.global_vars = IndentedBuffer()
        # Reduction data structure
        self.reduction_epilogue_suffix = IndentedBuffer()
        self.reduction_fusion = False
        self.reduction_body_loop = None
        self.reduction_idx = None
        self.reduction_buffer_idx = 0
        self.reduction_info = {}
        self.reduction_epilogue_result = {}
        self.reduction_mean = []
        self.reuse_buffer_names = {}

        # Overwrite ops
        self.load = self.load_epilogue
        self.store = self.store_epilogue
        self.store_reduction = self.store_reduction_epilogue
        self.reduction = self.reduction_epilogue

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

    def gemm_combination_mapping(self, M, N, K, n_extra_node=0, n_prologue_node=0, pad_k=True, min_tile=False):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer
        force_double_buffer = 2 if n_extra_node > 0 else 1 # In fusion case, double buffer should be forced
        minimum_n_tile = self.num_cores * force_double_buffer if min_tile else 1
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
            for i in tile_M_range:
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node) + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    n_tile = math.ceil(M / tile_M) * math.ceil(N / tile_N)
                    check_spad_size = (used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size and maximize_i_j <= tile_M * tile_N and n_tile >= minimum_n_tile and tile_N // tile_M < 10:
                        max_used_spad_size = used_spad_size
                        maximize_i_j = tile_M * tile_N
                        mapping = (tile_M, tile_N, tile_K)
        return mapping

    def search_mapping_space(self, mapping, idx, increment, stride, dilation, n_extra_node=0):
        if idx == 0 or idx == 1 or idx == 4 or idx == 5 or idx == 6:
            raise NotImplementedError("Only O_H and O_W are supported for search_mapping_space")
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer

        mapping = list(mapping)
        mapping[idx] += increment
        k_h, k_w, o_h, o_w, M, N, K = mapping
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
        if used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane:
            mapping = (k_h, k_w, o_h, o_w, M, N, K)
        else:
            mapping[idx] -= increment

        return mapping

    def pseudo_auto_tune(self, mapping, stride, dilation, O_H, O_W, n_extra_node=0):
        # pseudo auto-tune
        if mapping[2] == 1 and not (O_H == 1):
            mapping = self.search_mapping_space(mapping, 2, 1, stride, dilation, n_extra_node=n_extra_node)
        if mapping[3] == 1 and not (O_W == 1):
            mapping = self.search_mapping_space(mapping, 3, 1, stride, dilation, n_extra_node=n_extra_node)
        return mapping

    def conv_combination_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 # double buffer

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node, pad_k=False)
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
                        if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size and used_spad_size_per_lane < max_spad_per_lane and max_k_h_w <= k_h * k_w and max_o_h_w <= o_h * o_w:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h * k_w
                            max_o_h_w = o_h * o_w
                            mapping = (k_h, k_w, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")

        # FIXME: this should be implemented with auto-tuning
        mapping = self.pseudo_auto_tune(mapping, stride, dilation, O_H, O_W, n_extra_node=n_extra_node)

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
            kernel_name if self.outer_func_name is None else self.outer_func_name + f"_{len(call_args)}",
            call_args, cuda=False)

    def codegen_prologue_body(self):
        with self.prologue_buffer_group.as_local() as buf:
            buf.body.splice(buf.spad_buffer)
            buf.body.splice(buf.applys)
            buf.body.splice(buf.dma_loads)

            if (buf.loads.getvalue() != '' or buf.compute.getvalue() != '' or buf.stores.getvalue() != ''):
                buf.body.writelines(self.prologue_compute_body_loop.lines())
                compute_body = mlir_common.ParallelLoopBuffer()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                    compute_body.splice(buf.loads)
                    compute_body.splice(buf.compute)
                    compute_body.splice(buf.stores)
                buf.body.splice(compute_body)

        # Clear buffers
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def codegen_epilogue_body(self):
        def template_store():
            zero_cse = self.get_const_cse(0)
            sram_var = self.epilogue_info["sram_var"]
            dram_var = self.epilogue_info["dram_var"]
            index_var = self.epilogue_info["index_var"]
            tag_var = self.epilogue_info["tag_var"]
            mlir_dtype = self.epilogue_info["mlir_dtype"]
            dram_shape = self.epilogue_info["dram_shape"]
            vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis
            vlane_stride = self.kernel_group.tile_desc.get_vlane_stride()
            tile_stride = self.epilogue_info["tile_stride"]
            tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
            sram_index_var = ",".join([f"%{zero_cse}"] *  self.kernel_group.tile_desc.get_nr_dim())
            code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 tag_var, dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_stores, code, assignment = False)
        # Do dma store first to overlap epilogue nodes
        if self.reduction_fusion:
            if len(self.stores._lines) == 0:
                template_store()
                self.epilogue_buffer_group.body.splice(self.dma_stores)
                self.dma_stores.clear()
        self.epilogue_buffer_group.body.splice(self.spad_buffer)
        self.epilogue_buffer_group.body.splice(self.applys)
        self.epilogue_buffer_group.body.splice(self.dma_loads)
        self.epilogue_buffer_group.body.writelines(self.compute_body_loop.lines())
        compute_body = mlir_common.ParallelLoopBuffer()
        with contextlib.ExitStack() as stack:
            stack.enter_context(compute_body.indent(attribute="{inner_loop=false}",suffix=self.compute_body_loop.epilogue_line()))
            if self.reduction_fusion:
                #if len(self.stores._lines) == 0:
                #    template_store()
                compute_body.writelines(self.reduction_body_loop.lines())
                stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                compute_body.splice(self.loads)
                compute_body.splice(self.compute)
            else:
                compute_body.splice(self.loads)
                compute_body.splice(self.compute)
                if len(self.stores._lines) == 0:
                    template_store()
            compute_body.splice(self.epilogue_buffer_group.stores)
        if (compute_body.getvalue()):
            self.epilogue_buffer_group.body.splice(compute_body)
        self.epilogue_buffer_group.body.splice(self.dma_stores)
        self.epilogue_buffer_group.body.splice(self.reduction_epilogue_suffix)

        # Clear buffers
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

    def prepare_input(self, indent_size: int = 0):
        def emit_dma_start(buffer_name, index_var, tag_var, size, tile_size, subtile_size=None, async_flag=True, label="X"):
            base = f"memref.dma_start %{label}[%{index_var}], %{buffer_name}[%c0, %c0], %c_mvin"
            if label == "W":
                base = base.replace("mvin", "mvin2")

            suffix = f"%{tag_var}[%c0], %axis, %vstride"
            memref_shape = f"memref<{size}xf32>"
            tile_shape = "x".join([str(x) for x in tile_size])
            tile_memref = f"memref<{tile_shape}xf32, 1>"
            tag_memref = f"memref<1xi32>"
            attrs = f"sram_stride=[1, {tile_size[0]}]"
            async_flag = "false"
            if subtile_size:
                subtile_shape = ", ".join([str(x) for x in subtile_size])
                attrs = f"subtile_size=[{subtile_shape}], async={async_flag}, {attrs}"
            else:
                subtile_shape = ", ".join([str(x) for x in tile_size])
                attrs = f"subtile_size=[{subtile_shape}], async={async_flag}, {attrs}"
            attr_memref = f"{{ {attrs} }}"
            return f"{base}, {suffix}: {memref_shape}, {tile_memref}, {tag_memref} {attr_memref}"

        def hook():
            code = IndentedBuffer()
            self.codegen_prologue_body()
            prologue_code = self.prologue_buffer_group.body
            if prologue_code.getvalue():
                code.writeline(emit_dma_start(self.prologue_info["input_sram_var"], self.prologue_info["input_index_var"], self.prologue_info["input_tag_var"],
                                              self.prologue_info["input_numel"], self.prologue_info["input_tile_size"], subtile_size=self.prologue_info["input_subtile_size"], label="X"))
                code.splice(prologue_code)
                code.writeline(emit_dma_start(self.prologue_info["weight_sram_var"], self.prologue_info["weight_index_var"], self.prologue_info["weight_tag_var"],
                                              self.prologue_info["weight_numel"], self.prologue_info["weight_tile_size"], subtile_size=self.prologue_info["weight_subtile_size"], label="W"))
            else:
                code.writeline(emit_dma_start(self.prologue_info["input_sram_var"], self.prologue_info["input_index_var"], self.prologue_info["input_tag_var"],
                                              self.prologue_info["input_numel"], self.prologue_info["input_tile_size"], self.prologue_info["input_subtile_size"], async_flag=True, label="X"))
                code.writeline(emit_dma_start(self.prologue_info["weight_sram_var"], self.prologue_info["weight_index_var"], self.prologue_info["weight_tag_var"],
                                              self.prologue_info["weight_numel"], self.prologue_info["weight_tile_size"], self.prologue_info["weight_subtile_size"], async_flag=True, label="W"))
            code = textwrap.indent(code.getvalue(), " "*indent_size).strip()
            return code

        assert "<PREPARE_INPUT>" not in self.render_hooks
        self.render_hooks["<PREPARE_INPUT>"] = hook
        return "<PREPARE_INPUT>"

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
            self.codegen_epilogue_body()
            return textwrap.indent(self.epilogue_buffer_group.body.getvalue(), " "*indent_size).strip()

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

    def reduction_iter_arg(self):
        def hook():
            if len(self.reduction_vars):
                args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
                dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
                return f"iter_args({args}) -> ({dtype})"
            return ""

        assert "<REDUCTION_ITER_ARG>" not in self.render_hooks
        self.render_hooks["<REDUCTION_ITER_ARG>"] = hook
        return "<REDUCTION_ITER_ARG>"

    def reduction_acc(self):
        def hook():
            if len(self.reduction_vars):
                acc = ', '.join([f"%{acc.name}" for acc in self.reduction_vars.keys()])
                return f"{acc} ="
            return ""

        assert "<REDUCTION_ACC>" not in self.render_hooks
        self.render_hooks["<REDUCTION_ACC>"] = hook
        return "<REDUCTION_ACC>"

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

    def load_prologue(self, name: str, index: sympy.Expr):
        load_dim = []
        if not isinstance(V.graph, NullHandler) and name in V.graph.graph_inputs:
            load_dim = V.graph.graph_inputs[name].layout.size
        if self.ranges == self.buffer_types[name][2]:
            index_var = self.prologue_info['input_index_var'] if len(load_dim) != 1 else 'tile_n'
            vlane_split_axis = self.kernel_group.prologue_tile_desc.vlane_split_axis if len(load_dim) != 1 else 0    # FIXME: Fixed split axis for 1d load dim
        else:
            # Broadcast pattern
            zero_index = self.const_cse.generate(self.const_buffer, "arith.constant 0 : index")
            if self.prologue_info['is_bmm']: # FIXME: hardcoded
                idx = f"%b, %t_k, %t_n"
                map_var = self.map_cse.generate(self.global_vars, f"affine_map<(d0, d1, d2) -> (d0 * 512 + d2)>")
                vlane_split_axis = 2 # 3D GEMM prologue should be loaded by axis 2
            else:
                idx = f"%t_m, %{zero_index}"
                map_var = self.map_cse.generate(self.global_vars, f"affine_map<(d0, d1) -> (d0)>")
                vlane_split_axis = 1 # 2D GEMM prologue should be loaded by axis 1
            index_var = self.apply_cse.generate(self.dma_loads, f"affine.apply #{map_var}({idx})")
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.input(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_stride = self.kernel_group.prologue_tile_desc.vlane_stride if len(load_dim) != 1 else 1    # FIXME: Fixed stride for 1d load dim
        tile_numel_per_lane = self.kernel_group.prologue_tile_desc.get_numel_per_lane()
        tile_shape = self.kernel_group.prologue_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.prologue_info['input_sram_stride']

        # Compute vector unit size
        vshape = self.kernel_group.prologue_tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.prologue_tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index, self.alloc_buffer)
            self.buffer_names[name] = sram_var
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     f"{name}_tag", dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_loads, code, assignment = False)

        # Load vector from sram
        sram_var = self.buffer_names[name]
        zero_var = self.get_const_cse(0)
        compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.prologue_tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])

        if compute_vec_size > 1:
            operation = "affine.vector_load"
            line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.load"
            line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"

        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [compute_vec_size, mlir_dtype])
        return out

    def store_prologue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        tile_shape = self.kernel_group.prologue_tile_desc.get_mlir_shape(mlir_dtype)

        # Compute vector unit size
        vshape = self.kernel_group.prologue_tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.prologue_tile_desc.get_compute_vec_size()

        sram_var = self.buffer_names[name]
        zero_var = self.get_const_cse(0)

        _, operand_type = self.var_info[value]
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype, var_info=self.var_info)
        compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.prologue_tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
        # Generate vector load instruction
        if compute_vec_size > 1:
            operation = "affine.vector_store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}"
        self.stores.writeline(line)

    def load_epilogue(self, name: str, index: sympy.Expr):
        is_1d_source = len(index.free_symbols) == 1
        is_transpose = False    # FIXME: Only works for 2d input
        if len(index.args) == 2:
            for expr in index.args:
                if len(expr.args):
                    if expr.args[1].name == "index0" and expr.args[0] > 1:
                        is_transpose = True
                        break
        key = 't_index_var' if is_transpose else 'index_var'
        index_var = self.epilogue_info[key] if not is_1d_source else 'tile_n'
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.input(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis if not is_1d_source else 0    # FIXME: Fixed split axis for 1d load dim
        vlane_stride = self.kernel_group.tile_desc.vlane_stride if not is_1d_source else 1    # FIXME: Fixed stride for 1d load dim
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.epilogue_info['tile_stride']

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)
            self.buffer_names[name] = sram_var
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     f"{name}_tag", dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_loads, code, assignment = False)
        elif name in self.reuse_buffer_names:
            sram_var = self.reuse_buffer_names[name]
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     f"{name}_tag", dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.dma_loads, code, assignment = False)
        else:
            sram_var = self.buffer_names[name]

        # Load vector from sram
        zero_var = self.get_const_cse(0)
        if not self.reduction_fusion:
            compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
            if compute_vec_size > 1:
                operation = "affine.vector_load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
            else:
                operation = "affine.load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"
            out = self.cse.generate(self.loads, line)
            self.register_var_info(out, [compute_vec_size, mlir_dtype])
        else: # For reduction case
            reduce_size = self.reduction_nr_outer_loop
            vsize = compute_vec_size//reduce_size
            vshape = f"vector<{vsize}x{mlir_dtype}>"

            if compute_vec_size > 1:
                offset = self.cse.generate(self.loads, f"affine.apply affine_map<(d0, d1) -> (d0 + d1*{(self.reduction_axis_size)})>(%{self.compute_idx}, %{self.reduction_loop_idx})")
                compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{offset}"])
                operation = "affine.vector_load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
                out = self.cse.generate(self.loads, line)
            else:
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"
                out = self.cse.generate(self.loads, line)
            self.register_var_info(out, [self.compute_body_loop.step, mlir_dtype])
        return out

    def store_epilogue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index_var = self.epilogue_info['index_var']
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vlane_stride
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.epilogue_info['tile_stride']

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)
            self.buffer_names[name] = sram_var
        else:
            zero_cse = self.get_const_cse(0)
            sram_dims = len(tile_shape.split("x")) - 1
            sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)
        sram_var = self.buffer_names[name]
        zero_var = self.get_const_cse(0)

        _, operand_type = self.var_info[value]
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype, var_info=self.var_info)
        compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
        # Generate vector load instruction
        if compute_vec_size > 1:
            operation = "affine.vector_store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}"
        self.stores.writeline(DeferredLine(name, line))

        # Generate DMA instruction
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
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
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        vec_size = self.compute_body_loop.step
        vshape = f"vector<{vec_size}x{type_name}>"

        tile_shape = f"memref<{self.reduction_body_loop.size * self.vector_lane}x{vec_size}x{type_name}, 1>"
        name = f"{reduction_type}_buffer{self.reduction_buffer_idx}"
        self.reduction_buffer_idx += 1
        index = "dummy_index" # Not used
        tile_numel_per_lane = self.compute_body_loop.step * self.reduction_body_loop.size
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, None, index, self.const_buffer)
        self.reduction_epilogue_result[reduction_key] = sram_var

        # Load partial result
        zero_var = self.get_const_cse(0)
        operation = "affine.vector_load"
        compute_index_var = ",".join([f"%{self.reduction_loop_idx}"] + [f"%{zero_var}"])
        line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [self.compute_body_loop.step, type_name])

        # Reduction body codegen
        result = reduction_partial_combine_vec(reduction_type, value, out)

        # Store partial result
        operation = "affine.vector_store"
        line = f"{operation} %{result}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        self.compute.writeline(line) # Need to be placed after partial reduction
        self.reduction_info[sram_var] = reduction_type
        return sram_var

    def store_reduction_epilogue(self, name, index, value):
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        index = self.rename_indexing(index)

        # Tile is always reuduced in inner loop
        numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()
        reduction_axis_size = self.kernel_group.tile_desc.get_tile_size()[-2]
        nr_outer_loop = numel_per_lane // reduction_axis_size

        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis - 1
        vlane_stride = self.kernel_group.tile_desc.vlane_stride
        tile_numel_per_lane = vlane_stride * nr_outer_loop * 2

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = f"memref<{self.kernel_group.tile_desc.get_tile_size()[1]}x{type_name}, 1>"
        tile_stride = [1]
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index,
                                                                         index, buffer=self.const_buffer)
        for i in range(self.reduction_body_loop.size):
            vec_size = self.compute_body_loop.step
            vshape = f"vector<{vec_size}x{type_name}>"

            partial_tile_shape = f"memref<{self.reduction_body_loop.size * self.vector_lane}x{vec_size}x{type_name}, 1>"
            # Load partial result
            init = self.const_cse.generate(self.const_buffer, f"arith.constant {reduction_init(self.reduction_info[value], dtype)} : {type_name}")
            init_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{init} : {type_name} to {vshape}")
            zero_var = self.const_cse.generate(self.const_buffer, f"arith.constant {0} : index")
            index_var = self.const_cse.generate(self.const_buffer, f"arith.constant {i} : index")
            compute_index_var = ",".join([f"%{index_var}"] + [f"%{zero_var}"])

            operation = "affine.vector_load"
            line = f"{operation} %{value}[{compute_index_var}] : {partial_tile_shape}, {vshape}"
            out = self.cse.generate(self.reductions_suffix, line)
            operation = "affine.vector_store"
            line = f"{operation} %{init_vec}, %{value}[{compute_index_var}] : {partial_tile_shape}, {vshape}"
            self.reductions_suffix.writeline(line)

            # 2 step reduction
            new_vec_size = 2
            new_vshape = f"vector<{vec_size//new_vec_size}x{new_vec_size}x{type_name}>"
            new_reduced_shape = f"vector<{new_vec_size}x{type_name}>"
            out = self.cse.generate(self.reductions_suffix, f"vector.shape_cast %{out} : {vshape} to {new_vshape}")
            init_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{init} : {type_name} to {new_reduced_shape}")
            out = self.cse.generate(self.reductions_suffix, reduction_combine_vec(self.reduction_info[value], out, init_vec, axis=0, shape=new_vshape, reduced_shape=new_reduced_shape))
            out2 = self.cse.generate(self.reductions_suffix, f"vector.shuffle %{out}, %{out} [1, 0] : {new_reduced_shape}, {new_reduced_shape}")

            self.compute, self.reductions_suffix = self.reductions_suffix, self.compute
            self.register_var_info(out, [new_vec_size, type_name])
            self.register_var_info(out2, [new_vec_size, type_name])
            out = reduction_partial_combine_vec(self.reduction_info[value], out, out2)
            self.compute, self.reductions_suffix = self.reductions_suffix, self.compute

            # Final reduction
            #final_reduced_shape = type_name
            #init = self.const_cse.generate(self.const_buffer, f"arith.constant {reduction_init(self.reduction_info[value], dtype)} : {type_name}")
            #out = self.cse.generate(self.reductions_suffix, reduction_combine_vec(self.reduction_info[value], out, init, axis=0, shape=vshape, reduced_shape=final_reduced_shape))

            if self.welford_reduce_out is not None:
                # mean
                divider = self.cse.generate(self.reductions_suffix, f"arith.constant {float(768)} : f32")
                if self.buffer_types[name][1] > 1:
                    divider_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %{divider} : f32 to {new_reduced_shape}")
                else:
                    divider_vec = divider

                if self.current_node.node.origin_node: # FIXME: This is a temporary solution
                    # mean = E(X) / N
                    self.reduction_mean.append(self.cse.generate(self.reductions_suffix, f"arith.divf %{out}, %{divider_vec} : {new_reduced_shape}"))
                    out = self.reduction_mean[i]
                else:
                    # m2 = (E(X^2) - E(X)^2) * N
                    sqr_mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{out}, %{divider_vec} : {new_reduced_shape}")
                    mean_sqr = self.cse.generate(self.reductions_suffix, f"arith.mulf %{self.reduction_mean[i]}, %{self.reduction_mean[i]} : {new_reduced_shape}")
                    variance = self.cse.generate(self.reductions_suffix, f"arith.subf %{sqr_mean}, %{mean_sqr} : {new_reduced_shape}")
                    m2 = self.cse.generate(self.reductions_suffix, f"arith.mulf %{variance}, %{divider_vec} : {new_reduced_shape}")
                    out = m2

            operation = "affine.vector_store"
            line = f"{operation} %{out}, %{sram_var}[%{index_var}] : {tile_shape}, {new_reduced_shape}"
            self.reductions_suffix.writeline(DeferredLine(name, line))

        # MVOUT Encoding
        # Generate DMA instruction
        index_var = self.reduction_idx
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, type_name, dram_var, index_var, sram_var, sram_index_var,
                                f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.reductions_suffix.writeline(DeferredLine(name, code))

    def get_scratchpad_buffer(self, dtype, name, tile_size_per_lane, dram_tile_shape, index_var, raw_index, buffer=None):
        return super().get_scratchpad_buffer(dtype, name, tile_size_per_lane, dram_tile_shape, index_var, raw_index, True, buffer=buffer)

    def set_tile_size(self, template_epilogue_info, prologue=False):
        tile_desc = mlir_common.MLIRMultiDimTile(template_epilogue_info['tile_size'],
            self.vector_lane,
            vlane_split_axis=template_epilogue_info['vlane_split_axis'],
            vlane_stride=template_epilogue_info['vlane_stride'])

        if "reuse_buffer_names" in template_epilogue_info:
            self.reuse_buffer_names.update(template_epilogue_info["reuse_buffer_names"])

        if 'nr_rdim' in template_epilogue_info and template_epilogue_info['nr_rdim']==1:
            tile_desc.nr_rdim = 1
            numel_per_lane = tile_desc.get_numel_per_lane()
            reduction_axis_size = tile_desc.get_tile_size()[-2]
            nr_outer_loop = (numel_per_lane + reduction_axis_size-1) // reduction_axis_size
            tile_desc.vec_size = nr_outer_loop * 32 # Why? Emprically selected, other option failed to functionality...

            self.reduction_fusion = True
            self.reduction_axis_size =  tile_desc.get_tile_size()[-2]
            self.reduction_nr_outer_loop = (numel_per_lane + reduction_axis_size-1) // reduction_axis_size
            self.reduction_idx = template_epilogue_info["reduction_idx"]
            self.reduction_loop_idx = "reduce_loop_idx"
            self.compute_body_loop.size = reduction_axis_size
            self.compute_body_loop.step = tile_desc.get_compute_vec_size() // nr_outer_loop
            self.reduction_body_loop = mlir_common.LoopLevel(self.reduction_loop_idx, nr_outer_loop)
        else:
            tile_desc.vec_size=64
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