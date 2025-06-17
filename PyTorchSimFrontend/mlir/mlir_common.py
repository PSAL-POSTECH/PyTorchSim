import dataclasses
import math
from typing import Dict
from typing import List
from collections import defaultdict
from functools import reduce
from operator import mul
import torch
from torch._dynamo.testing import rand_strided
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codegen import common
from torch._inductor.codegen import cpp
from torch._inductor.virtualized import V
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
from torch.utils._sympy.functions import ModularIndexing
import sympy
import contextlib

from typing import Callable

import sympy

import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from torch._inductor.utils import (
    free_symbol_startswith,
    get_sympy_Expr_dtype,
    IndentedBuffer,
    sympy_subs,
    sympy_symbol,
    unique,
)
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

DTYPE_TO_MLIR = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "i8",
    torch.bool: "i8",
    torch.bfloat16: "bf16",
}

DTYPE_TO_C = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "int64_t",
    torch.int32: "int32_t",
    torch.int16: "int16_t",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.bool: "uint8_t",
    torch.bfloat16: "bfloat16",
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]

MLIR_INF = {
    "inf" : {
        "f32" : 0x7F800000,
        "f64" : 0x7FF0000000000000
    },
    "-inf" : {
        "f32" : 0xFF800000,
        "f64" : 0xFFF0000000000000
    },
    "nan" : {
        "f32" : 0x7FC00000,
        "f64" : 0x7FF8000000000000
    }
}

class ParallelLoopBuffer(IndentedBuffer):
    def indent(self, offset=1, attribute="", suffix=""):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                if suffix:
                    self.writeline(suffix)
                self._indent -= 1
                self.writeline("} " + attribute)
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                if suffix:
                    self.writeline(suffix)
                self._indent -= 1
                self.writeline("} " + attribute)

        return ctx()

class MLIRKernelArgs(common.KernelArgs):
    MLIR_ARGS_IN = 0x01
    MLIR_ARGS_OUT = 0x02
    MLIR_ARGS_INOUT = 0x04
    MLIR_ARGS_VAR = 0x08

    def __init__(self, tile_row=None, tile_col=None):
        super().__init__()
        self.tile_row = tile_row
        self.tile_col = tile_col

    @staticmethod
    def is_mlir_arg_in(value):
        return (MLIRKernelArgs.MLIR_ARGS_IN & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    @staticmethod
    def is_mlir_arg_out(value):
        return (MLIRKernelArgs.MLIR_ARGS_OUT & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    @staticmethod
    def is_mlir_arg_inout(value):
        return MLIRKernelArgs.MLIR_ARGS_INOUT & value

    @staticmethod
    def get_mlir_shape(info):
        tensor_type = DTYPE_TO_MLIR[info[0]]
        return f"memref<{info[1]}x{tensor_type}>"

    def mlir_argdefs(self, extra_node=dict()):
        buffer_types = {}
        for x in V.graph.buffers:
            if not isinstance(x.layout, MultiOutputLayout): # FIXME: MultiOutputLayout should be handled
                buffer_types[x.get_name()] = [x.get_dtype(), x.get_numel(), x.get_size(), x.get_stride()]
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                buffer_types[name] = [get_sympy_Expr_dtype(val), 1, [1], [1]]
            else:
                buffer_types[name] = [val.get_dtype(), val.get_numel(), val.get_size(), val.get_stride()]
        buffer_types.update(
            {name: [val.dtype, 1, [1], [1]] for name, val in V.graph.constants.items()}
        )
        buffer_types.update(
            {name: [val.get_dtype(), val.get_numel(), val.get_size(), val.get_stride()] for name, val in extra_node.items()}
        )

        call_args = []
        arg_defs = []
        arg_attributes = []
        def set_info(outer, inner, arg_type):
            mlir_shape = self.get_mlir_shape(buffer_types[outer])
            arg_defs.append(f"%{inner}: {mlir_shape}")
            call_args.append(outer)
            arg_attributes.append([outer] + [[arg_type] + buffer_types[outer]])

        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            set_info(outer, inner, self.MLIR_ARGS_INOUT)
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            set_info(outer, inner, self.MLIR_ARGS_IN)
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            set_info(outer, inner, self.MLIR_ARGS_OUT)
        for outer, inner in self.sizevars.items():
            set_info(outer, inner, self.MLIR_ARGS_VAR)
        return arg_defs, call_args, arg_attributes, buffer_types

class MLIRMultiDimTile():
    def __init__(self, tile_size, vector_lane, vlane_split_axis=None, vlane_stride=None, vec_size=None):
        self._tile_size = list(tile_size)
        self.tile_axis_order = list(range(len(tile_size)))
        self.vec_size = vec_size

        # Vector lane mapping config
        self.vector_lane = vector_lane
        self.vlane_split_axis = vlane_split_axis
        self.vlane_stride = vlane_stride
        self.implicit_dim_size = None
        self.nr_rdim = 0

    def set_tile_size(self, tile_size, tile_axis_order=None):
        self._tile_size = tile_size
        if tile_axis_order is None:
            self.tile_axis_order = list(range(len(tile_size)))
        else:
            self.tile_axis_order = tile_axis_order

    def get_tile_size(self):
        return self._tile_size

    def get_numel(self):
        """
        Return size of multi-dimensional tile
        """
        size = 1
        for dim_size in self._tile_size:
            size *= dim_size
        return size

    def get_numel_per_lane(self):
        tile_size_per_lane = self.get_tile_size_per_lane()
        size = 1
        for dim_size in tile_size_per_lane:
            size *= dim_size
        return size

    def get_tile_stride(self):
        strides = [1] * len(self._tile_size)
        init = 1

        original_indices = list(range(len(self.tile_axis_order)))
        sorted_pairs = sorted(
            zip(self.tile_axis_order, self._tile_size, original_indices),
            key=lambda x: x[0], reverse=True
        )
        for _, size, original_indices in sorted_pairs:
            strides[original_indices] = init
            init *= size
        return strides

    def get_tile_size_per_lane(self):
        tile_size_per_lane = list(self._tile_size)
        if self.vlane_split_axis < 0 or self.vlane_split_axis >= len(tile_size_per_lane):
            raise AssertionError("Not allowed split_axis")
        used_vlane = self.get_used_vlane()
        tile_size_per_lane[self.vlane_split_axis] = \
            self.div_round_up(tile_size_per_lane[self.vlane_split_axis], used_vlane)
        return tile_size_per_lane

    def get_nr_dim(self):
        """
        Return number of dimensions
        """
        return len(self._tile_size)

    def get_dim_size(self, index):
        if isinstance(index, int):
            return self._tile_size[index]
        elif "index" in str(index):
            return self._tile_size[int(str(index)[5:])]
        raise NotImplementedError("Unsupported format of index")

    def get_mlir_shape(self, dtype):
        str_tile_size = [str(dim) for dim in self._tile_size]
        shape = "x".join(str_tile_size)
        return f"memref<{shape}x{dtype}, 1>"

    def get_mlir_vshape(self, mlir_dtype):
        return f"vector<{self.get_compute_vec_size()}x{mlir_dtype}>" if self.get_compute_vec_size() > 1 else f"{mlir_dtype}"

    def get_used_vlane(self):
        """
        Return number of used vector lane
        """
        if self.vlane_split_axis < 0 or self.vlane_split_axis >= len(self._tile_size):
            raise AssertionError("Not allowed split_axis")
        return min(self.div_round_up(self._tile_size[self.vlane_split_axis], self.vlane_stride), self.vector_lane)

    def get_vlane_stride(self):
        return self.vlane_stride

    def get_compute_vec_size(self):
        # Granule size used in compute loop
        if self.vec_size is not None:
            return self.vec_size
        if self.nr_rdim:
            assert self.nr_rdim==1
            val = self.get_numel_per_lane() // self._tile_size[-1]
            if self.get_numel_per_lane() >= val * 8:
                return val*8
            elif self.get_numel_per_lane() >= val * 4:
                return val*4
            elif self.get_numel_per_lane() >= val * 2:
                return val*2
            return val
        if (self.get_numel_per_lane() // self.vlane_stride) >= 8:
            return self.vlane_stride * 8
        if (self.get_numel_per_lane() // self.vlane_stride) >= 4:
            return self.vlane_stride * 4
        if (self.get_numel_per_lane() // self.vlane_stride) >= 2:
            return self.vlane_stride * 2
        return self.vlane_stride

    @staticmethod
    def div_round_up(size, round_val):
        return (size + round_val - 1) // round_val

class MLIRWrapperKenrelGroup(cpp.KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = MLIRKernelArgs()
        self.tile_desc : MLIRMultiDimTile = None

    def set_tile_info(self, tile_desc : MLIRMultiDimTile):
        self.tile_desc = tile_desc

    def set_prologue_tile_info(self, tile_desc : MLIRMultiDimTile):
        self.prologue_tile_desc = tile_desc

class BaseMLIRHardwareInfo():
    def __init__(self):
        # Default HW setting
        self.vector_lane = extension_config.CONFIG_VECTOR_LANE
        self.spad_info = extension_config.CONFIG_SPAD_INFO
        self.precision = extension_config.CONFIG_PRECISION
        self.num_cores = extension_config.CONFIG_NUM_CORES
        self.vlen = extension_config.CONFIG_VLEN

class BaseMLIRKernel(common.Kernel, BaseMLIRHardwareInfo):
    newvar_prefix = "%"
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, kernel_group, reason=None):
        super().__init__(kernel_group.args)
        self.kernel_group = kernel_group
        # Kernel iteration range info
        self.call_ranges = None
        self.ranges = None
        self.reduction_depth = None
        self.itervars = None
        # Code buffer
        self.vector_compute = IndentedBuffer()
        self.reductions_suffix = IndentedBuffer()
        self.cse = common.CSE(self.newvar_prefix, self.suffix)
        # MLIR SSA tracker
        self.var_info = {} # MLIR variable info
        self.buffer_types : dict = None # format: dtype, numel, size, stride
        self.compute_idx = "compute_idx"
        self.compute_body_loop = LoopLevel(self.compute_idx, 1)
        self.recodegen = reason # spad overflow, tile size, vlane stride
        self.stop_autotune = False

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"index{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def store(self, name, index, value, mode=None):
        raise NotImplementedError()

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def indirect_indexing(self, index_var, size, check):
        raise NotImplementedError()

    def codegen_global_init(self):
        raise NotImplementedError()

    def codegen_loops(self):
        raise NotImplementedError()

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def is_modular_indexing(self, expr):
        return "ModularIndexing" in str(expr)

    def compute_tile_size(self, nodes, vars, reduction_vars):
        # Handle implict dims. Input operand could have larger dimension space.
        implicit_ranges = False
        target_operand : MemoryDep = None
        implicit_dim_size = defaultdict(list)
        for read_operand in nodes[0].read_writes.reads:
            read_operand : MemoryDep
            if isinstance(read_operand, StarDep) or isinstance(read_operand, WeakDep): # FIXME: WeakDep & StarDep are not supported (MoE case)
                continue
            read_index = read_operand.index
            for arg in read_index.args:
                if "ModularIndexing" in str(arg) or "//" in str(arg):
                    implicit_ranges = True
                    target_operand = read_operand
                    break

        if implicit_ranges:
            #print("This operation contain implicit dimension space!")
            linearized_stride = [1] * len(target_operand.var_names)
            for i in range(len(target_operand[3])-2, -1, -1):
                linearized_stride[i] = linearized_stride[i+1] * target_operand[3][i+1]

            linearized_index = sympy.Integer(0)
            for dim, stride in zip(target_operand[2], linearized_stride):
                linearized_index += stride * dim

            new_dim_expression = []
            new_dim_size = []
            for arg in target_operand.index.args:
                if len(arg.free_symbols) != 1:
                    raise NotImplementedError("Not supporting this view operation...!")

                if arg.is_Mul and arg.args[0].is_number:
                    arg = arg.args[1]

                if isinstance(arg, ModularIndexing):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], arg.args[2])
                elif arg.is_symbol:
                    modular_expr = ModularIndexing(arg, 1, target_operand.ranges[arg])
                elif "//" in str(arg):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], target_operand.ranges[arg.args[0]]//arg.args[1])
                else:
                    raise NotImplementedError("What is this case?")
                new_dim_expression.append(modular_expr)
                new_dim_size.append(modular_expr.args[2])
                implicit_dim_size[int(str(modular_expr.args[0])[1:])].append(int(modular_expr.args[2]))

            # Sanity check
            for dim, sub_dims in implicit_dim_size.items():
                sz = reduce(mul, sub_dims, 1)
                if sz != target_operand[3][dim]:
                    raise NotImplementedError("Not supporting type...")

        vlane_split_axis = len(vars) - 1 # Set split_axis as a last normal loop not reduction loop

        # FIXME: Naive decrease tile size
        def decrease_tile_size(tile_size, vlane_split_axis):
            is_decreased = False

            # Decrease vlane_split_axis when it is too large
            if tile_size[vlane_split_axis] > vlane_stride * self.vector_lane:
                tile_size[vlane_split_axis] = int(tile_size[vlane_split_axis] // 2)
                return tile_size

            for i in range(len(tile_size)):
                if i == vlane_split_axis:
                    continue
                if tile_size[i] > 1:
                    tile_size[i] = int(tile_size[i] // 2)
                    is_decreased = True
                    break

            # Decrease vlane_split_axis at the end to maximize the vlane usage
            if not is_decreased:
                if tile_size[vlane_split_axis] > 1:
                    tile_size[vlane_split_axis] = int(tile_size[vlane_split_axis] // 2)
            return tile_size

        # Dummy tile size
        def dummy_tile_size():
            tile_size = [1] * (len(vars) + len(reduction_vars))
            if len(tile_size) == 2:
                tile_size[-1] = vlane_stride * self.vector_lane
                tile_size[-2] = 2 * self.vector_lane
            elif len(tile_size) == 0: # Scalar
                tile_size = [1]
                self.ranges = [1]
            elif len(tile_size) == 1:
                tile_size[0] = 2 * vlane_stride * self.vector_lane
            elif len(tile_size) == 3:
                tile_size[-1] = self.vector_lane
                tile_size[-2] = 2 * self.vector_lane
                tile_size[-3] = 2
            else:
                raise NotImplementedError("dummy tile size fail!")
            return tile_size

        vlane_stride = extension_config.CONFIG_VECTOR_LANE_STRIDE
        if self.recodegen is None:
            tile_size = dummy_tile_size()
        else:
            if self.recodegen == "spad_overflow":
                tile_size = self.kernel_group.tile_desc.get_tile_size()
                decrease_tile_size(tile_size, vlane_split_axis)
            elif self.recodegen == "vlane_stride":
                tile_size = dummy_tile_size()
            elif "tile_size" in self.recodegen:
                dim = int(self.recodegen.split("_")[-1])
                tile_size = self.kernel_group.tile_desc.get_tile_size() # TODO:
                tile_size[dim] = tile_size[dim] * 2
            else:
                raise NotImplementedError(f"Unknown recodegen reason: {self.recodegen}")

        # FIXME: Not considering removed buffers
        n_buffer = sum(
            len(node.read_writes.reads) + len(node.read_writes.writes)
            for node in nodes
        )

        spad_overflow = True
        # Find proper tile size
        while spad_overflow:
            # Adjust tile size to avoid too much paddings
            for i in range(1, len(tile_size)+1):
                target_range = self.ranges[-i]
                if implicit_ranges:
                    target_range = implicit_dim_size[len(tile_size)-i][-1]

                if tile_size[-i] > target_range:
                    remains = (target_range % vlane_stride)
                    self.stop_autotune = True
                    tile_size[-i] = target_range
                    if remains:
                        tile_size[-i] += vlane_stride - remains

            # Adjust tile size
            for i in range(len(vars)):
                if tile_size[i] >= self.vector_lane: # maximize used vector lane
                    vlane_split_axis = i
            used_vlane = min((tile_size[vlane_split_axis] + vlane_stride - 1) // vlane_stride, self.vector_lane)
            padded_size = used_vlane * vlane_stride
            tile_size[vlane_split_axis] = ((tile_size[vlane_split_axis] + padded_size - 1) // padded_size) * padded_size

            # Check spad overflow
            spad_usage_per_vlane = n_buffer * math.prod(tile_size) * self.precision // used_vlane
            if spad_usage_per_vlane >= self.spad_info["spad_size"]:
                new_tile_size = decrease_tile_size(tile_size.copy(), vlane_split_axis)
                if new_tile_size == tile_size:
                    raise NotImplementedError("Error: Cannot find proper tile size")
                tile_size = new_tile_size
                spad_overflow = True
                self.stop_autotune = True # for auto-tune
                continue
            else:
                spad_overflow = False

        # Maximize the utilizaiotn of vectorlane
        if len(reduction_vars):
            minimum_stride = max(self.roundup_vectorlane(tile_size[vlane_split_axis]) // self.vector_lane, 2)
            vlane_stride = min(minimum_stride, 8)

        # Handle scalar case
        if len(self.ranges)==1 and self.ranges[0] == 1:
            vlane_stride = 1
            vlane_split_axis = 0
            tile_size[0] = 1
        elif vlane_split_axis == -1:
            vlane_split_axis = 0
            vlane_stride = tile_size[0]

        # Select tile info.
        # Note: Kernel Group have to share same tile desc for fusion
        tile_desc = MLIRMultiDimTile(tile_size, self.vector_lane)
        tile_desc.vlane_split_axis = vlane_split_axis
        tile_desc.vlane_stride = vlane_stride
        tile_desc.implicit_dim_size = implicit_dim_size
        tile_desc.nr_rdim = len(reduction_vars)
        return tile_desc

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        # Set node range info
        vars, reduction_vars = self.set_ranges(group, reduction_group)
        tile_desc = self.compute_tile_size(nodes, vars, reduction_vars)
        self.compute_body_loop.size = tile_desc.get_numel_per_lane()
        self.compute_body_loop.step = tile_desc.get_compute_vec_size()
        self.kernel_group.set_tile_info(tile_desc)

        _, _, _, self.buffer_types = self.kernel_group.args.mlir_argdefs()
        with self as kernel:
            for node in nodes:
                node.run(vars, reduction_vars)
        V.graph.removed_buffers |= self.removed_buffers
        # V.graph.inplaced_to_remove |= self.inplaced_to_remove
        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()
        return src_code

    def run_bench(self, nodes, kernel_name, src_code):
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        input_call_args = tuple(self.args.input_buffers.keys())
        output_call_args = tuple(self.args.output_buffers.keys())
        full_input_nodes = tuple([V.graph.get_buffer(k) for k in input_call_args])
        full_output_nodes = tuple([V.graph.get_buffer(k) for k in output_call_args])

        bmreq = MLIRBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(full_output_nodes),
            extra_args={
                "vector_lane" : self.vector_lane,
                "spad_info": self.spad_info,
                "vlen" : self.vlen,
                "arg_attributes" : arg_attributes
            },
            source_code=src_code,
        )
        dummy_inputs = [rand_strided(meta.sizes,meta.strides,dtype=meta.dtype, extra_size=meta.offset).to(device=nodes[0].get_device()) for meta in bmreq.input_tensor_meta]
        dummy_outputs = [rand_strided(meta.sizes,meta.strides,dtype=meta.dtype, extra_size=meta.offset).to(device=nodes[0].get_device()) for meta in bmreq.output_tensor_meta]
        return bmreq.make_run_fn(dummy_inputs, dummy_outputs)

    def codegen_kernel(self, kernel_name):
        arg_defs, _, _, _ = self.kernel_group.args.mlir_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        #TODO:. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"

        code.splice(self.codegen_global_init())
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.kernel_group.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        # Dump loop and load/store information
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")
        return arg_attributes

    def get_constant_vector(self, expr):
        constant_vector = [[int(expr.coeff(var)),None] for var in self.itervars]
        return constant_vector

    def get_constant_vector2(self, expr):
        # Case 0. symbol ex) index 0
        # Case 1. inner product form ex) 16 * index0 + 1 * index1
        # Case 2. Complicated form ex) 16 * index0 + 8 * (index//4) + (index % 4)
        constant_vector = []
        if expr.is_symbol:
            constant_vector.append(tuple([1, expr]))
            return constant_vector

        for arg in expr.args:
            if arg.is_symbol:
                constant_vector.append(tuple([1,arg]))
                continue
            if len(arg.args) == 0: #TODO: check this
                continue
            if arg.args[0].is_number:
                constant_vector.append(arg.args)
            else:
                constant_vector.append([1, arg])

        return constant_vector

    def find_node_by_name(self, name):
        if name in V.graph.graph_inputs:
            return V.graph.graph_inputs[name]
        else:
            for output_node in V.graph.graph_outputs:
                if output_node.data.name == name:
                    return output_node

    def is_scalar(self, name):
        return self.buffer_types[name][1] == 1

    def roundup_vectorlane(self, size, amp=1):
        return ((size + self.vector_lane - 1) // self.vector_lane) * self.vector_lane * amp

    def register_var_info(self, var, var_info):
        self.var_info[var] = var_info

    def rename_indexing(self, index) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.kernel_group.args.size(x)
            for x in sorted_symbols
            if x.name.startswith("s") or x.name.startswith("ps")
        }
        return sympy_subs(index, replacements)

    def __enter__(self):
        class CSEProxy:
            self.name = "CSEProxy"

            @staticmethod
            def __getattr__(name: str) -> Callable[..., common.CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    code, ret_info = getattr(parent_handler, name)(*args, var_info=self.var_info)
                    csevar = self.cse.generate(
                        self.compute,
                        code,
                        bounds=ValueRanges.unknown(),
                        assignment=(ret_info[0] is not None)
                    )
                    if ret_info[0] is not None:
                        self.register_var_info(csevar, ret_info)
                        csevar.update_on_args(name, args, kwargs)
                    return csevar

                return inner

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                # Skip CSE since this doesn't return an expression
                return self.indirect_indexing(index_var, size, check)

            @staticmethod
            def load(name: str, index: sympy.Expr):
                if name in self.cse.invalidated_stores:
                    # A load from an invalidated store requires us to
                    # keep the actual buffer around
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_startswith(index, "%"):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return store_cache[name]
                key = name+str(index)
                if key not in self.cse.cache:
                    result = self.load(name, index)
                    self.cse.cache[key] = result
                return self.cse.cache[key]

            @staticmethod
            def store(name, index, value, mode=None):
                self.store_buffer_names.add(name)
                if mode is None:
                    self.cse.store_cache[name] = value
                    if self.current_node:
                        for other_name in self.current_node.get_mutations():
                            self.cse.store_cache[other_name] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)

            @staticmethod
            def store_reduction(name, index, value):
                self.store_buffer_names.add(name)
                self.cse.store_cache[name] = value
                if self.current_node:
                    for other_name in self.current_node.get_mutations():
                        self.cse.store_cache[other_name] = value

                if name not in V.graph.removed_buffers:
                    return self.store_reduction(name, index, value)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def _index_expr(tile_size, buffer, renamed_expression, index):
                return self._index_expr(tile_size, buffer, renamed_expression, index)

            @staticmethod
            def index_expr(index, dtype):
                return self.index_expr(index, dtype)

            @staticmethod
            def bucketize(
                values,
                offsets_name: str,
                offsets_size: sympy.Expr,
                indexing_dtype: torch.dtype,
                right: bool,
            ):
                """
                [Note: Inductor bucketize op]

                Given values (tensor) and offsets_name (reference to the name of a 1D
                tensor), calculate the bucket that each value belongs to.

                e.g. for values [-1, 0, 1, 2, 3, 4, 5, 9], offsets [0, 4, 4, 8], right=True
                return =        [ 0, 1, 1, 1, 1, 3, 3, 4].

                When right == False, bucket i refers to range (offsets[i], offsets[i+1]].
                When right == True,  bucket i refers to range [offsets[i], offsets[i+1]).

                Offsets must be non-decreasing or the result is undefined.
                """
                return self.bucketize(
                    values, offsets_name, offsets_size, indexing_dtype, right
                )

        super().__enter__()
        assert self.overrides
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    start: int = 0
    step: int = 1
    reduction_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    affine_yield: Dict[str, str] = dataclasses.field(default_factory=dict)

    def lines(self):
        if len(self.reduction_vars):
            acc = ', '.join([f"%{acc.name}" for acc in self.reduction_vars.keys()])
            args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
            dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
            line = f"{acc} = affine.for %{self.var} = {self.start} to {self.size} step {self.step} iter_args({args}) -> ({dtype})"
        else:
            line = f"affine.for %{self.var} = {self.start} to {self.size} step {self.step}"

        return [line]

    def epilogue_line(self):
        if len(self.affine_yield):
            vars = ', '.join([f"%{name}" for name, _ in self.affine_yield.items()])
            reduced_shapes = ', '.join([f"{shape}" for _, shape in self.affine_yield.items()])
            return f"affine.yield {vars} : {reduced_shapes}"
        return ""

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars, affine_yield=dict()):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars
            loop.affine_yield = affine_yield

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd