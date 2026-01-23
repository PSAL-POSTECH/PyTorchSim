import dataclasses
import math
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict
from typing import List
from collections import defaultdict
from functools import reduce
from operator import mul
import torch
from torch._inductor.codegen import common
from torch._inductor.codegen import cpp
from torch._inductor.virtualized import V
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
from torch._inductor.codegen.wrapper import KernelDefinitionLine
from torch.utils._sympy.functions import ModularIndexing, FloorDiv, Mod, Identity
import sympy
import contextlib

from typing import Callable

import sympy

from torch.utils._sympy.value_ranges import ValueRanges
from torch._inductor.utils import (
    get_sympy_Expr_dtype,
    IndentedBuffer,
    sympy_subs,
    unique,
)
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend import extension_codecache

from PyTorchSimFrontend.extension_utils import (
    free_symbol_startswith,
    sympy_symbol
)

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

MLIR_TO_DTYPE = {
    "f32": torch.float32,
    "f64": torch.float64,
    "f16": torch.float16,
    "i64": torch.int64,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8":  torch.int8,
    "bf16": torch.bfloat16,
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
    torch.bfloat16: "uint16_t",
}

MLIR_TO_BIT = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "bf16": 16,
    "index": 64
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

class RecompileSignal(BaseException):
    """
    Exception raised when a recompilation of a kernel or code block is required.
    """
    def __init__(self, message="Recompilation requested."):
        self.message = message
        super().__init__(self.message)

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

class VectorLaneMapping():
    def __init__(self, vector_lane: int, forced_vec_size: int, vlane_split_axis: int, vlane_stride: int):
        self.vector_lane = vector_lane
        self.vlane_split_axis = vlane_split_axis
        self.vlane_stride = vlane_stride
        self.forced_vec_size = forced_vec_size

    def get_used_vlane(self, tile_size: list[int]):
        return min(
            math.ceil(tile_size[self.vlane_split_axis] / self.vlane_stride),
            self.vector_lane
        )

    def get_tile_size_per_lane(self, tile_size: list[int]):
        per_lane = tile_size.copy()
        used = self.get_used_vlane(tile_size)
        if self.vlane_split_axis < 0 or self.vlane_split_axis >= len(per_lane):
            raise AssertionError("Not allowed split_axis")
        per_lane[self.vlane_split_axis] = math.ceil(per_lane[self.vlane_split_axis] / used)
        return per_lane

    def get_numel_per_lane(self, tile_size: list[int]):
        return math.prod(self.get_tile_size_per_lane(tile_size))

    def get_tile_stride_per_lane(self, tile_size: list[int], tile_stride: list[int]):
        tile_stride = tile_stride.copy()  # original strides
        get_tile_size_per_lane = self.get_tile_size_per_lane(tile_size)
        coeff = tile_size[self.vlane_split_axis]//get_tile_size_per_lane[self.vlane_split_axis]

        # Propagate stride according to per-lane tile size
        for i in range(len(tile_stride)):
            if tile_stride[i] > tile_stride[self.vlane_split_axis]:
                tile_stride[i] = tile_stride[i] // coeff
        return tile_stride

    def get_compute_vec_size(self, tile_size: list[int], reduction_numel: int, nr_rdim: int) -> int:
        if self.forced_vec_size is not None:
            return self.forced_vec_size

        per_lane = self.get_numel_per_lane(tile_size)
        stride = self.vlane_stride
        if nr_rdim:
            val = per_lane // max(reduction_numel, 1)
            for mult in [8, 4, 2]:
                if per_lane >= val * mult:
                    return val * mult
            return val
        for mult in [8, 4, 2]:
            if (per_lane // stride) >= mult:
                return stride * mult
        return stride

class TileAdjustMixin():
    def __init__(self):
        self.tail_ratio_threshold = 0.01

    def apply_divisor(self, axis: int, divisor: int, mode: str = "split"):
        """Split or pad a given axis of the tile."""
        old_size = self._tile_size[axis]
        if divisor <= 1:
            return

        padded = math.ceil(old_size / divisor) * divisor
        outer = math.ceil(old_size / divisor)
        inner = divisor

        if mode == "pad":
            self._tile_size[axis] = padded
            self.update_tile_stride()
            return
        elif mode == "split":
            new_sizes = list(self._tile_size)
            new_sizes[axis] = outer
            new_sizes.insert(axis + 1, inner)
            self._tile_size = new_sizes

            old_order_val = self.tile_axis_order[axis]
            new_order = list(self.tile_axis_order)
            new_order.insert(axis + 1, old_order_val + 0.1)
            self.tile_axis_order = [idx for idx, _ in sorted(
                zip(range(len(new_order)), new_order), key=lambda x: x[1]
            )]
            self.update_tile_stride()

            # Adjust split axis for vmap
            if self.vmap.vlane_split_axis > axis:
                self.vmap.vlane_split_axis += 1
            return

        raise ValueError(f"Unknown mode: {mode}. Supported: 'pad', 'split'.")

    def is_dim_dividable(self, dim_sizes: list[int]) -> bool:
        if len(dim_sizes) != len(self._tile_size):
            raise ValueError("dim_sizes must match the tile size dimensions")

        dim_sizes_cpy = list(dim_sizes)
        axis, stride = self.vmap.vlane_split_axis, self.vmap.vlane_stride
        remain = dim_sizes_cpy[axis] % stride
        if remain:
            dim_sizes_cpy[axis] += stride - remain

        return all(d % t == 0 for d, t in zip(dim_sizes_cpy, self._tile_size))

    def adjust_tile_to_divisible(self, dim_sizes: list[int]) -> list[int]:
        """Adjust current tile to be divisible by given dimensions."""
        if len(dim_sizes) != len(self._tile_size):
            raise ValueError("dim_sizes must match the tile size dimensions")

        def _adjust_one(dim_size, tile_size):
            for candidate in range(tile_size, 0, -1):
                if dim_size % candidate == 0:
                    return candidate
            return 1

        candidate_tile_size = [_adjust_one(d, t) for d, t in zip(dim_sizes, self._tile_size)]
        for i in range(len(candidate_tile_size)):
            self.tile_constraint[i].must_divide_dim = True

        axis, stride = self.vmap.vlane_split_axis, self.vmap.vlane_stride
        remain = candidate_tile_size[axis] % stride

        if remain:
            # #201: relax vlane_stride constraints
            self.vmap.vlane_stride = 1
        return candidate_tile_size

    def scale_tile_dim(self, axis, dim_sz, scale_factor=2):
        axis_constrinat = self.tile_constraint[axis]
        current_sz = self._tile_size[axis]
        new_sz = axis_constrinat.adjust(current_sz, int(current_sz * scale_factor), dim_sz)
        self._tile_size[axis] = new_sz
        self.update_tile_stride()
        return current_sz != new_sz

    def decrease_tile_size(self, dim_size):
        tile_size = self._tile_size
        vlane_split_axis, vlane_stride, vector_lane = self.vmap.vlane_split_axis, self.vmap.vlane_stride, self.vmap.vector_lane
        tile_size = list(tile_size)

        # Decrease vlane_split_axis when it is too large
        if tile_size[vlane_split_axis] > 2 * vlane_stride * vector_lane:
            if self.scale_tile_dim(vlane_split_axis, dim_size[vlane_split_axis], scale_factor=0.5):
                return

        for i in range(len(tile_size)):
            if i == vlane_split_axis:
                continue
            if tile_size[i] > 1:
                if self.scale_tile_dim(i, dim_size[i], scale_factor=0.5):
                    return

        # Decrease vlane_split_axis at the end to maximize the vlane usage
        self.scale_tile_dim(vlane_split_axis, dim_size[vlane_split_axis], scale_factor=0.5)
        return

    def trim_large_tail(self, ranges: list[int]):
        for i, (dim_range, tile_range) in enumerate(zip(ranges, self._tile_size)):
            ALPHA = 1.0
            BETA = 0.5
            constraint = self.tile_constraint[i]
            if constraint.fixed:
                continue
            elif constraint.must_divide_dim:
                BETA = 0

            padding_ratio = TileAdjustMixin.get_padding_ratio(tile_range, dim_range)
            if padding_ratio < self.tail_ratio_threshold:
                continue
            best_tile = tile_range
            best_cost = (
                ALPHA * padding_ratio +
                BETA * (dim_range / tile_range)
            )

            min_tile = 1
            for candidate in range(tile_range - 1, min_tile - 1, -1):
                new_candidate = constraint.adjust(tile_range, candidate, dim_range)
                ratio = TileAdjustMixin.get_padding_ratio(new_candidate, dim_range)
                iter_penalty = (dim_range / new_candidate)

                cost = ALPHA * ratio + BETA * iter_penalty
                if cost < best_cost:
                    best_tile, best_cost = new_candidate, cost
            self._tile_size[i] = best_tile

    def select_vlane_axis(self):
        best_vlane_split_axis = 0
        best_used_vlane = math.ceil(self._tile_size[0] / self.vmap.vlane_stride)
        for i, dim in enumerate(self._tile_size[1:len(self._tile_size)-self.nr_rdim]):
            used_vlane = math.ceil(dim / self.vmap.vlane_stride)
            if used_vlane > best_used_vlane:
                best_used_vlane = used_vlane
                best_vlane_split_axis = i+1
        self.vmap.vlane_split_axis = best_vlane_split_axis

    def pad_vlane_tile(self):
        # FIXME. this doesn't follow tile constraints...
        vlane_split_axis, vlane_stride, vector_lane = self.vmap.vlane_split_axis, self.vmap.vlane_stride, self.vmap.vector_lane
        used_vlane = min(math.ceil(self._tile_size[vlane_split_axis] / vlane_stride), vector_lane)
        padded_size = used_vlane * vlane_stride
        self._tile_size[vlane_split_axis] = math.ceil(self._tile_size[vlane_split_axis] / padded_size) * padded_size

    def apply_constraints(self, constraints, ranges):
        for idx, (axis_constraints, axis_size) in enumerate(zip(constraints.values(), ranges)):
            for const in axis_constraints:
                if const.args[1] == 1:
                    continue
                divider = int(const.args[1])

                if not self.tile_constraint[idx].fixed:
                    self.tile_constraint[idx].fixed = True
                    self._tile_size[idx] = divider
                elif self.tile_constraint[idx].fixed and self._tile_size[idx] > divider:
                    self._tile_size[idx] = divider
        self.update_tile_stride()

    @staticmethod
    def init_tile_size(ranges, vlane_stride, vector_lane):
        nr_dim = len(ranges)
        tile_size = [1] * nr_dim
        if len(tile_size) == 2:
            tile_size[-1] = vlane_stride * vector_lane
            tile_size[-2] = 2 * vector_lane
        elif len(tile_size) == 0: # Scalar
            tile_size = [1]
            ranges = [1]
        elif len(tile_size) == 1 and ranges[0]==1:
            tile_size[0] = 1
        elif len(tile_size) == 1:
            tile_size[0] = 2 * vlane_stride * vector_lane
        elif len(tile_size) == 3:
            tile_size[-1] = vector_lane
            tile_size[-2] = 4 * vector_lane
            tile_size[-3] = 2
        elif len(tile_size) == 4:
            tile_size[-1] = vector_lane
            tile_size[-2] = 4 * vector_lane
            tile_size[-3] = 2
            tile_size[-4] = 1
        else:
            raise NotImplementedError("dummy tile size fail!")
        return tile_size

    @staticmethod
    def get_padding_ratio(tile_range: int, dim_range: int) -> float:
        if tile_range <= 0 or dim_range <= 0:
            raise ValueError("tile_range and dim_range must be positive integers")
        tail = dim_range % tile_range
        padding = (tile_range - tail) % tile_range
        return float(padding / dim_range)

@dataclass
class TileConstraint:
    multiple_of: int = 1
    must_divide_dim: bool = False
    fixed: bool = False

    def adjust(self, old: int, new: int, dim: int) -> int:
        if self.fixed:
            return old # Fixed tile size

        tail = new % self.multiple_of
        new -= tail
        if not self.must_divide_dim:
            return max(new, self.multiple_of)

        while new > 0:
            if dim % new == 0:
                return new
            new -= self.multiple_of
        raise extension_codecache.TileSizeError("Cannot find suitable tile size under the given constraints.")

class MLIRMultiDimTile(TileAdjustMixin):
    def __init__(self, tile_size, vector_lane, vlane_split_axis=None, vlane_stride=None, forced_vec_size=None):
        super().__init__()
        self.name = ""
        self._tile_size = list(tile_size)
        self._tile_stride = None
        self.tile_constraint = [TileConstraint(vlane_stride if idx == vlane_split_axis else 1) for idx, _ in enumerate(tile_size)]
        self.tile_axis_order = list(range(len(tile_size)))
        self.update_tile_stride()

        # Vector lane mapping config
        self.vmap = VectorLaneMapping(
            vector_lane=vector_lane,
            forced_vec_size=forced_vec_size,
            vlane_split_axis=vlane_split_axis,
            vlane_stride=vlane_stride
        )

        self.implicit_dim_size = None
        self.nr_rdim = 0
        self.offset = sympy.Integer(0) # Dram offset

    def set_name(self, name: str): self.name = name
    def get_name(self) -> str: return self.name
    def get_tile_size(self): return list(self._tile_size)
    def get_tile_stride(self): return list(self._tile_stride)
    def get_numel(self) -> int :return math.prod(self._tile_size)
    def get_nr_dim(self) -> str: return len(self._tile_size)
    def get_reduction_numel(self): return reduce(mul, self.get_tile_size()[-1*self.nr_rdim:], 1)

    def set_tile_size(self, tile_size, tile_axis_order=None, constraints=None):
        self._tile_size = list(tile_size)
        self.tile_axis_order = list(range(len(tile_size))) if tile_axis_order is None else tile_axis_order
        self.update_tile_stride()

    def set_tile_size_stride(self, tile_size, tile_stride):
        self._tile_size = list(tile_size)
        self._tile_stride = list(tile_stride)

    def update_tile_stride(self):
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
        self._tile_stride = strides

    def get_dim_size(self, index):
        if isinstance(index, int):
            return self._tile_size[index]
        elif "index" in str(index):
            return self._tile_size[int(str(index)[5:])]
        raise NotImplementedError("Unsupported format of index")

   # Vector mapping delegation
    def get_tile_size_per_lane(self): return self.vmap.get_tile_size_per_lane(self._tile_size)
    def get_used_vlane(self): return self.vmap.get_used_vlane(self._tile_size)
    def get_numel_per_lane(self): return self.vmap.get_numel_per_lane(self._tile_size)
    def get_tile_stride_per_lane(self): return self.vmap.get_tile_stride_per_lane(self._tile_size, self._tile_stride)
    def get_compute_vec_size(self): return self.vmap.get_compute_vec_size(self._tile_size, self.get_reduction_numel(), self.nr_rdim)

    # Helper functions for codegen
    def get_mlir_shape(self, dtype):
        shape = "x".join([str(dim) for dim in self._tile_size])
        return f"memref<{shape}x{dtype}, 1>"

    def get_mlir_vshape(self, mlir_dtype):
        return f"vector<{self.get_compute_vec_size()}x{mlir_dtype}>" if self.get_compute_vec_size() > 1 else f"{mlir_dtype}"

class MLIRWrapperKenrelGroup(cpp.KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = MLIRKernelArgs()
        self.tile_desc : MLIRMultiDimTile = None

    def set_tile_info(self, tile_desc : MLIRMultiDimTile):
        self.tile_desc = tile_desc

class BaseMLIRHardwareInfo():
    def __init__(self):
        # Default HW setting
        self.vector_lane = extension_config.vpu_num_lanes
        self.spad_info = extension_config.CONFIG_SPAD_INFO
        self.precision = extension_config.CONFIG_PRECISION
        self.num_cores = extension_config.CONFIG_NUM_CORES
        self.vlen = extension_config.vpu_vector_length_bits

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
        self.itervar_cses = None
        # Code buffer
        self.vector_compute = IndentedBuffer()
        self.reductions_suffix = IndentedBuffer()
        self.cse = common.CSE(self.newvar_prefix, self.suffix)
        # MLIR SSA tracker
        self.var_info = {} # MLIR variable info
        self.buffer_types : dict = None # format: dtype, numel, size, stride
        # Create compute idx
        self.compute_idx = self.register_var_cse("compute_idx", 1, "index")
        self.compute_body_loop = LoopLevel(self.compute_idx, 1)
        self.prologue_compute_body_loop = LoopLevel(self.compute_idx, 1)
        self.recodegen = reason # spad overflow, tile size, vlane stride
        self.stop_autotune = False

        instance_id = id(self)
        self.target_buffer_override = contextvars.ContextVar(f"Handler_compute_override_{instance_id}", default=self.compute)
        self.target_cse_override = contextvars.ContextVar(f"Handler_cse_override_{instance_id}", default=self.cse)
        self._nested_context_depth = 0

    def set_ranges(self, lengths, reduction_lengths, index_names=None):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            if index_names is None:
                self.itervars = [sympy.Symbol(f"index{n}") for n in range(len(self.ranges))]
            else:
                assert len(index_names) == len(self.ranges), f"Index names length mismatch: {len(index_names)} != {len(self.ranges)}"
                self.itervars = [sympy.Symbol(str(n)) for n in index_names]

            self.itervar_cses = {str(index) : self.register_var_cse(str(index), 1, "index") for index in self.itervars}
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def get_nr_rdim(self):
        return len(self.itervars[self.reduction_depth:])

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def store(self, name, index, value, mode=None):
        raise NotImplementedError()

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def indirect_indexing(self, index_var, size, check, wrap_neg):
        raise NotImplementedError()

    def codegen_global_init(self):
        raise NotImplementedError()

    def codegen_loops(self):
        raise NotImplementedError()

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, triton=False)

    def is_modular_indexing(self, expr):
        return "ModularIndexing" in str(expr)

    def implicit_dim_ops(self, nodes):
        target_patterns = (ModularIndexing, FloorDiv, Mod)
        target_operands = []
        for target_node in nodes:
            for read_operand in target_node.read_writes.reads:
                read_operand: MemoryDep
                if isinstance(read_operand, StarDep) or isinstance(read_operand, WeakDep):
                    continue
                read_index = read_operand.index
                for arg_expr in read_index.args:
                    if arg_expr.atoms(*target_patterns):
                        target_operands.append(read_operand)
        return target_operands

    def extract_dividers(self, implicit_ops):
        # When a specific axis is processed, the key constraint to verify is the divider.
        # The tile size must be forced to match the divider size.
        dim_dividers = defaultdict(set)
        for operand in implicit_ops:
            subs_map = {
                s: sympy.symbols(s.name.replace("c", "index", 1))
                for s in operand.index.free_symbols
            }
            rev_subs_map = {
                sympy.symbols(s.name.replace("c", "index", 1)) : s
                for s in operand.index.free_symbols
            }
            new_index = operand.index.subs(subs_map)
            for arg in new_index.args:
                if arg.is_number:
                    continue
                if len(arg.free_symbols) > 1:
                    raise NotImplementedError("Not supporting this view operation...!")
                if arg.is_Mul and arg.args[0].is_number:
                    arg = arg.args[1]

                if isinstance(arg, ModularIndexing):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], arg.args[2])
                    modular_expr.original_expr = arg
                elif arg.is_symbol:
                    modular_expr = ModularIndexing(arg, 1, operand.ranges[rev_subs_map[arg]])
                    modular_expr.original_expr = arg
                elif "//" in str(arg):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], operand.ranges[rev_subs_map[arg.args[0]]]//arg.args[1])
                    modular_expr.original_expr = arg
                else:
                    raise NotImplementedError("What is this case?")
                dim_dividers[modular_expr.args[0]].add(modular_expr)
        return dim_dividers

    def compute_tile_size(self, nodes, vars, reduction_vars):
        vlane_split_axis = len(vars) - 1
        vlane_stride = 2 # Set minimum vlane stride

        # Set initial tile size & vector lane mapping
        if self.kernel_group.tile_desc is None:
            tile_size = MLIRMultiDimTile.init_tile_size(self.ranges, vlane_stride, self.vector_lane)
            init_tile_desc = MLIRMultiDimTile(tile_size, self.vector_lane, vlane_split_axis, vlane_stride)
            init_tile_desc.nr_rdim = len(reduction_vars)
            self.kernel_group.set_tile_info(init_tile_desc)

            # Handle edge case
            if len(self.ranges)==1 and self.ranges[0] == 1: # Scalar case 2
                self.kernel_group.tile_desc.vmap.vlane_stride = 1
                self.kernel_group.tile_desc.vmap.vlane_split_axis = 0
            elif vlane_split_axis == -1: # Reduction only case
                self.kernel_group.tile_desc.vmap.vlane_split_axis = 0
                self.kernel_group.tile_desc.vmap.vlane_stride = self.kernel_group.tile_desc.get_tile_size()[0]

        # Handle implict dims. Input operand could be high dimension tensor.
        # Note: https://github.com/PSAL-POSTECH/PyTorchSim/issues/173
        implicit_ops = self.implicit_dim_ops(nodes)
        if implicit_ops:
            tile_constraints = self.extract_dividers(implicit_ops)
            self.kernel_group.tile_desc.apply_constraints(tile_constraints, self.ranges)
            self.kernel_group.tile_desc.implicit_dim_size = tile_constraints

        # Check recodegen reason
        if self.recodegen is not None:
            if self.recodegen == "spad_overflow":
                self.kernel_group.tile_desc.decrease_tile_size(self.ranges)
            elif self.recodegen == "recompile":
                return self.kernel_group.tile_desc
            else:
                raise NotImplementedError(f"Unknown recodegen reason: {self.recodegen}")

        # Adjust tile size & vector lane mapping
        self.kernel_group.tile_desc.trim_large_tail(self.ranges)
        self.kernel_group.tile_desc.select_vlane_axis()
        self.kernel_group.tile_desc.pad_vlane_tile()
        self.kernel_group.tile_desc.update_tile_stride()
        return self.kernel_group.tile_desc

    def codegen_nodes(self, nodes, kernel_name):
        recompile_try = 0
        max_retry_compile = 5
        while True:
            _, (group, reduction_group) = max(
                nodes, key=lambda x: int(x.is_reduction())
            ).group

            # Set node range info
            vars, reduction_vars = self.set_ranges(group, reduction_group)
            tile_desc = self.compute_tile_size(nodes, vars, reduction_vars)
            self.compute_body_loop.size = tile_desc.get_numel_per_lane()
            self.compute_body_loop.step = tile_desc.get_compute_vec_size()
            try:
                _, _, _, self.buffer_types = self.kernel_group.args.mlir_argdefs()
                with self as kernel:
                    for node in nodes:
                        node.run(vars, reduction_vars)
            except RecompileSignal as e:
                recompile_try += 1
                if recompile_try > max_retry_compile:
                    raise RuntimeError("Failed to compile kernel after multiple attempts.")
                # Retry compile nodes
                #print(f"Try recompile({recompile_try}/{max_retry_compile}). Reason: {e}")
                continue
            V.graph.removed_buffers |= self.removed_buffers
            # V.graph.inplaced_to_remove |= self.inplaced_to_remove
            src_code = self.codegen_kernel(kernel_name=kernel_name)
            meta_code = self.meta_kernel()
            return src_code, meta_code

    def codegen_kernel(self, kernel_name):
        arg_defs, _, _, _ = self.kernel_group.args.mlir_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        #TODO:. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"

        code.splice(self.codegen_global_init())
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            # Loop body part
            code.splice(self.codegen_loops())
        return code.getvalue()

    def meta_kernel(self):
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        meta_code = arg_attributes
        return meta_code

    def get_constant_vector(self, expr):
        constant_vector = [[int(expr.coeff(var)),None] for var in self.itervars]
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

    def register_var_cse(self, name, size, dtype):
        var = self.create_cse_var(name, ValueRanges.unknown())
        self.register_var_info(var, [size, dtype])
        return var

    def register_var_info(self, var, var_info):
        self.var_info[var] = var_info

    def rename_indexing(self, index) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]

        # FIXME. This is a temporary solution to remove Identity wrappers from index expression.
        # Remove Identity wrappers from index expression
        # Check if index itself is Identity
        if isinstance(index, Identity):
            index = index.args[0] if index.args else index

        # Replace Identity arguments with Identity.args[0]
        if hasattr(index, 'args') and len(index.args) > 0:
            for arg in index.args:
                if arg.is_Mul and arg.args[0].is_number and isinstance(arg.args[1], Identity):
                    index = index.replace(arg.args[1], arg.args[1].args[0] if arg.args[1].args else arg.args[1])
                if isinstance(arg, Identity):
                    index = index.replace(arg, arg.args[0] if arg.args else arg)

        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.kernel_group.args.size(x)
            for x in sorted_symbols
            if x.name.startswith("s") or x.name.startswith("ps")
        }
        return sympy_subs(index, replacements)

    @contextmanager
    def override_buffer_cse(self, *, buffer=None, cse=None):
        buffer_override = self.target_buffer_override
        cse_override = self.target_cse_override
        buffer_token = cse_token = None
        try:
            # Store tokens for proper restoration in nested contexts
            # contextvars.set() returns the previous value (token) which can be used for reset()
            if buffer is not None:
                buffer_token = buffer_override.set(buffer)
            if cse is not None:
                cse_token = cse_override.set(cse)
            yield self
        finally:
            # Restore using tokens - contextvars automatically handles nested contexts
            # Each level restores to its own previous value
            if cse_token is not None:
                cse_override.reset(cse_token)
            if buffer_token is not None:
                buffer_override.reset(buffer_token)

    def __enter__(self):
        class CSEProxy:
            self.name = "CSEProxy"

            @staticmethod
            def __getattr__(name: str) -> Callable[..., common.CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    code, ret_info = getattr(parent_handler, name)(*args, **kwargs)
                    target_buffer = self.target_buffer_override.get()
                    target_cse = self.target_cse_override.get()
                    if isinstance(code, common.DeferredLine):
                        target_buffer.writeline(code)
                        return None
                    else:
                        csevar = target_cse.generate(
                            target_buffer,
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
            def indirect_indexing(index_var, size, check=True, wrap_neg=True):
                # Skip CSE since this doesn't return an expression
                return self.indirect_indexing(index_var, size, check, wrap_neg)

            @staticmethod
            def load(name: str, index: sympy.Expr):
                index = self.rename_indexing(index)
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
                if key not in self.cse._cache:
                    result = self.load(name, index)
                    self.cse._cache[key] = result
                return self.cse._cache[key]

            @staticmethod
            def store(name, index, value, mode=None):
                self.store_buffer_names.add(name)
                if mode is None:
                    self.cse.store_cache[name] = value
                    if self.current_node:
                        for other_name in self.current_node.get_output(name).get_mutations():
                            self.cse.store_cache[other_name] = value
                if name not in V.graph.removed_buffers:
                    index = self.rename_indexing(index)
                    return self.store(name, index, value, mode=mode)

            @staticmethod
            def store_reduction(name, index, value):
                self.store_buffer_names.add(name)
                self.cse.store_cache[name] = value
                if self.current_node:
                    for other_name in self.current_node.get_output(name).get_mutations():
                        self.cse.store_cache[other_name] = value

                if name not in V.graph.removed_buffers:
                    index = self.rename_indexing(index)
                    return self.store_reduction(name, index, value)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def _index_expr(tile_size, buffer, renamed_expression, index):
                return self._index_expr(tile_size, buffer, renamed_expression, index)

            @staticmethod
            def index_expr(index, dtype):
                index = self.rename_indexing(index)
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

        if self._nested_context_depth == 0:
            self.exit_stack.__enter__()
            assert self.overrides
            parent_handler = self.overrides()

            self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
            self.exit_stack.enter_context(V.set_kernel_handler(self))
        self._nested_context_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._nested_context_depth -= 1
        if self._nested_context_depth == 0:
            super().__exit__(exc_type, exc_val, exc_tb)

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
        for loop_depth, loop in enumerate(self.loops):
            loop.reduction_vars = {key: list(val)[:-1] for key, val in reduction_vars.items() if val[-1] == loop_depth}
            loop.affine_yield = {key: val[0] for key, val in affine_yield.items() if val[-1] == loop_depth}

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd