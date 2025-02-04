import functools
import itertools
import textwrap
import re
import math
import sympy

from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import Kernel, KernelTemplate, ChoiceCaller, OpOverrides, CSE
from torch._inductor.ir import Buffer, IRNode, TemplateBuffer
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V
from torch._inductor.utils import IndentedBuffer

from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo, MLIRTile
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

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
        self.render_hooks = dict()
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

    def gemm_combination_mapping(self, M, N, K):
        spad_size = self.spad_info["spad_size"] * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer
        M_padded = ((M + self.vector_lane - 1) // self.vector_lane) * self.vector_lane
        N_padded = ((N + self.vector_lane - 1) // self.vector_lane) * self.vector_lane
        K_padded = ((K + self.vector_lane - 1) // self.vector_lane) * self.vector_lane

        max_used_spad_size = 0
        mapping = (self.vector_lane, self.vector_lane, self.vector_lane)
        for tile_M in range(self.vector_lane, M_padded + 1, self.vector_lane):
            for tile_N in range(self.vector_lane, N_padded + 1, self.vector_lane):
                for tile_K in range(self.vector_lane, K_padded + 1, self.vector_lane):
                    used_spad_size = (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N) * self.precision
                    if used_spad_size < max_spad_size and max_used_spad_size < used_spad_size:
                        max_used_spad_size = used_spad_size
                        mapping = (tile_M, tile_N, tile_K)

        Outer_M = math.ceil(M_padded / mapping[0])
        Outer_N = math.ceil(N_padded / mapping[1])
        Outer_K = math.ceil(K_padded / mapping[2])

        # split mapping equally to avoid unnecessary padding
        mapping = (M_padded // Outer_M, N_padded // Outer_N, K_padded // Outer_K)
        return mapping

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        arg_attributes = self.kernel_arg_attributes
        if arg_attributes is None:
            _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
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

    def codegen_body(self, vlane_split_axis):
        def template_store(options):
            sram_var = "Y_buffer"
            dram_var = "Y"
            index_var = "index2"
            tag_var = "tag"
            vlane_stride = 1
            mlir_dtype = "f32"
            dram_shape = f"memref<{options['Y_numel']}x{mlir_dtype}>"
            tile_shape = f"memref<{options['TILE_M']}x{options['TILE_N']}x{mlir_dtype}, 1>"
            zero_cse = self.get_const_cse(0)
            sram_index_var = ",".join([f"%{zero_cse}"] * 2)
            tile_stride = [1, options['TILE_M']]
            code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 tag_var, dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.stores, code, assignment = False)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        if len(self.stores._lines) == 0:
            template_store(self.render_options)
        self.body.splice(self.stores)
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
                extra_node[node.get_name()] = node
                self.buffer_names[node.get_name()] = 'Y_buffer'   #TODO: Buffer name fixed

        def hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=extra_node)
            return f"({', '.join(arg_defs)})"

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

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

    def store_output(self, vlane_split_axis=1):
        def hook():
            self.codegen_body(vlane_split_axis)
            return textwrap.indent(self.body.getvalue(), "      ").strip()  #TODO: First line is not indented

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        return "<STORE_OUTPUT>"

    def def_function(self):
        _, call_args, _ = self.kernel_group.args.python_argdefs()
        if self.outer_func_render is not None:
            return self.outer_func_render(input_args=call_args)
        else:
            return None, None

    def def_global_vars(self):
        return "<GLOBAL_VARS>"

    def replace_global_vars(self):
        return textwrap.indent(self.global_vars.getvalue(), "").strip()

    def add_extra_global_vars(self, code):
        key = "<GLOBAL_VARS>"
        return code.replace(key, self.replace_global_vars())

    def def_local_vars(self):
        return "<LOCAL_VARS>"

    def replace_local_vars(self):
        code = IndentedBuffer()
        code.tabwidth = 2
        code.splice("\n")
        with code.indent():
            code.splice(self.const_buffer)
            code.splice(self.alloc_buffer)
        return code.getvalue()

    def add_extra_local_vars(self, code):
        key = "<LOCAL_VARS>"
        return code.replace(key, self.replace_local_vars())

    def render(self, template, kwargs):
        # self.render_hooks = {}
        return PartialRender(
            template.render(**kwargs),
            self.render_hooks,
        )

    def adjust_tile_size(self):
        # Fixed tile size for template kernel
        self.kernel_group.tile_desc.set_tile_size((self.render_options['TILE_M'], self.render_options['TILE_N']))
        self.kernel_group.tile_desc.vlane_split_axis = 1 # FIXME: Fixed
        self.kernel_group.tile_desc.vlane_stride = 1 # FIXME: Fixed
        return

    def load_epilogue(self, name: str, index: sympy.Expr):
        #index_var = self.parse_indices(index)
        index_var = "index2"
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.input(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vlane_stride
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()
        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
            # tile_stride = self.kernel_group.tile_desc.get_tile_stride()
            tile_stride = [1, self.render_options['TILE_M']] # FIXME: Fixed
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, self.loads, index_var, index)
            self.buffer_names[name] = sram_var
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     f"{name}_tag", dram_shape, tile_shape, tile_stride)
            self.cse.generate(self.loads, code, assignment = False)

        # Load vector from sram
        sram_var = self.buffer_names[name]
        operation = "affine.vector_load" if tile_numel_per_lane > 1 else "affine.load"
        shape = f", vector<{tile_numel_per_lane}x{mlir_dtype}>" if tile_numel_per_lane > 1 else ""
        zero_var = self.get_const_cse(0)
        line = f"{operation} %{sram_var}[%{zero_var}, %{zero_var}] : memref<{self.render_options['TILE_M']}x{self.render_options['TILE_N']}x{mlir_dtype}, 1>{shape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [tile_numel_per_lane, mlir_dtype])
        return out

    def store_epilogue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        #index_var = self.parse_indices(index)
        index_var = "index2"
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        vlane_split_axis = self.kernel_group.tile_desc.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vlane_stride
        tile_numel_per_lane = self.kernel_group.tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        # tile_stride = self.kernel_group.tile_desc.get_tile_stride()
        tile_stride = [1, self.render_options['TILE_M']] # FIXME: Fixed

        if name not in self.buffer_names:
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, self.stores, index_var, index)
            self.buffer_names[name] = sram_var
        sram_var = self.buffer_names[name]

        operation = "affine.vector_store" if tile_numel_per_lane > 1 else "affine.store"
        shape = f", vector<{tile_numel_per_lane}x{mlir_dtype}>" if tile_numel_per_lane > 1 else ""
        zero_var = self.get_const_cse(0)
        line = f"{operation} %{value}, %{sram_var}[%{zero_var}, %{zero_var}] : memref<{self.render_options['TILE_M']}x{self.render_options['TILE_N']}x{mlir_dtype}, 1>{shape}"
        self.cse.generate(self.stores, line, assignment = False)

        index_var = "index2"                # FIXME. Is it okay?
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.cse.generate(self.stores, code, assignment = False)

    def get_scratchpad_buffer(self, dtype, name, tile_size_per_lane, dram_tile_shape, code_buffer, index_var, raw_index):
        return super().get_scratchpad_buffer(dtype, name, tile_size_per_lane, dram_tile_shape, code_buffer, index_var, raw_index, True)

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