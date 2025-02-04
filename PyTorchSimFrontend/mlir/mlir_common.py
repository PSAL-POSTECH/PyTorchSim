import os
from collections import defaultdict
from functools import reduce
from operator import mul
import torch
from torch._inductor.codegen import common
from torch._inductor.codegen import cpp
from torch._inductor.virtualized import V
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.dependencies import MemoryDep
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
    torch.bool: "i1",
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

class ParallelLoopBuffer(IndentedBuffer):
    def indent(self, offset=1, outer_loop=True):
        @contextlib.contextmanager
        def ctx():
            attribute = "{outer_loop=true}" if outer_loop else "{accumulation_loop=true}"
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("} " + attribute)
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
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
    def __init__(self, tile_size, vector_lane, vlane_split_axis=None, vlane_stride=None):
        self._tile_size = list(tile_size)
        self.tile_axis_order = list(range(len(tile_size)))

        # Vector lane mapping config
        self.vector_lane = vector_lane
        self.vlane_split_axis = vlane_split_axis
        self.vlane_stride = vlane_stride
        self.implicit_dim_size = None

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

    def get_used_vlane(self):
        """
        Return number of used vector lane
        """
        return min(self.div_round_up(self._tile_size[self.vlane_split_axis], self.vlane_stride), self.vector_lane)

    def get_vlane_stride(self):
        return self.vlane_stride

    @staticmethod
    def div_round_up(size, round_val):
        return (size + round_val - 1) // round_val

class MLIRTile():
    TILE_ROW_WISE = 0
    TILE_COL_WISE = 1
    TILE_PER_LANE_ROW_WISE = 2
    TILE_PER_LANE_COL_WISE = 3
    def __init__(self, n_row, n_col, vector_lane, used_vector_lane=None) -> None:
        self.n_row = n_row
        self.n_col = n_col
        self.vector_lane = vector_lane
        if used_vector_lane is None:
            self.used_vector_lane = self.vector_lane
        else:
            self.used_vector_lane = used_vector_lane
        self.tile_per_lane_layout = self.TILE_PER_LANE_ROW_WISE # How a given tile per lane is stored
        self.tile_layout = self.TILE_ROW_WISE # How a given tile is stored per lane
        self.vector_lane_axis = (self.n_col//self.used_vector_lane) > 0 #(0: Col major, 1: Row major)

    def get_tile_size(self):
        return self.n_row * self.n_col

    def get_rows_per_lane(self):
        if self.n_row % self.used_vector_lane != 0 and self.n_row > 1:
            print(f"[Warning] n_row({self.n_row}) % vector_lane({self.used_vector_lane}) != 0")
        return self.div_round_up(self.n_row, self.used_vector_lane)

    def get_cols_per_lane(self):
        if self.n_col % self.used_vector_lane != 0 and self.n_col > 1:
            print(f"[Warning] n_col({self.n_col}) % vector_lane({self.used_vector_lane}) != 0")
        return self.div_round_up(self.n_col, self.used_vector_lane)

    def get_tile_size_per_lane(self):
        if self.get_tile_size() % self.used_vector_lane != 0:
            print(f"[Warning] n_col({self.n_col}) % vector_lane({self.used_vector_lane}) != 0")
        return self.div_round_up(self.get_tile_size(), self.used_vector_lane)

    def get_vlane_stride(self):
        if self.tile_layout == self.TILE_ROW_WISE:
            vlane_stride = self.get_tile_size_per_lane()
        else:
            vlane_stride = self.get_cols_per_lane()
        return vlane_stride

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

class BaseMLIRHardwareInfo():
    def __init__(self):
        # Default HW setting
        self.vector_lane = 128
        self.spad_info = {
            "spad_vaddr" : 0xD0000000,
            "spad_paddr" : 0xD0000000,
            "spad_size" : 128 << 10 # 128KB per Lane
        }
        self.precision = 4 # 32bit
        self.num_cores = 1
        self.vlen = 32 // self.precision # 256bits / 32bits = 8 [elements]

class BaseMLIRKernel(common.Kernel, BaseMLIRHardwareInfo):
    newvar_prefix = "%"
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, kernel_group):
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

    def codegen_global_init(self):
        raise NotImplementedError()

    def codegen_loops(self):
        raise NotImplementedError()

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def compute_tile_size(self, nodes, vars, reduction_vars):
        # Handle implict dims. Input operand could have larger dimension space.
        implicit_ranges = False
        target_operand : MemoryDep = None
        implicit_dim_size = defaultdict(list)
        for read_operand in nodes[0].read_writes.reads:
            read_operand : MemoryDep
            read_index = read_operand.index
            for arg in read_index.args:
                if "ModularIndexing" in str(arg) or "//" in str(arg):
                    implicit_ranges = True
                    target_operand = read_operand
                    break

        if implicit_ranges:
            print("This operation contina implicit dimension space!")
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

        # Dummy tile size
        tile_size = [1] * (len(vars) + len(reduction_vars))
        if len(tile_size) == 2:
            tile_size[-1] = 128
            tile_size[-2] = 128
        elif len(tile_size) == 0: # Scalar
            tile_size = [1]
            self.ranges = [1]
        elif len(tile_size) == 1:
            tile_size[0] = 128*128*2
        elif len(tile_size) == 3:
            tile_size[-1] = 128
            tile_size[-2] = 128
            tile_size[-3] = 2
        else:
            raise NotImplementedError("dummy tile size fail!")

        vlane_stride = 2
        # Adjust tile size to avoid too much paddings
        for i in range(1, len(tile_size)+1):
            target_range = self.ranges[-i]
            if implicit_ranges:
                target_range = implicit_dim_size[len(tile_size)-i][-1]

            if tile_size[-i] > target_range:
                remains = (target_range % vlane_stride)
                tile_size[-i] = target_range
                if remains:
                    tile_size[-i] += vlane_stride - remains

        # Select tile info.
        # Note: Kernel Group have to share same tile desc for fusion
        tile_desc = MLIRMultiDimTile(tile_size, self.vector_lane)
        tile_desc.vlane_split_axis = len(vars) - 1 # Set split_axis as a last normal loop not reduction loop
        tile_desc.vlane_stride = vlane_stride
        tile_desc.implicit_dim_size = implicit_dim_size
        return tile_desc

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        # Set node range info
        vars, reduction_vars = self.set_ranges(group, reduction_group)
        tile_desc = self.compute_tile_size(nodes, vars, reduction_vars)
        self.kernel_group.set_tile_info(tile_desc)

        _, _, _, self.buffer_types = self.kernel_group.args.mlir_argdefs()
        with self as kernel:
            for node in nodes:
                node.run(vars, reduction_vars)
        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()
        return src_code

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
        wrapper.add_import_once(f'\nfrom PyTorchSimFrontend.extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

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
                    # TritonTemplateKernel has no current_node
                    buf_bounds = ValueRanges.unknown()
                    if hasattr(V.interpreter, "current_node"):
                        fx_node = V.interpreter.current_node
                        assert isinstance(self.node_to_bounds, dict)
                        buf_bounds = self.node_to_bounds.get(
                            fx_node, ValueRanges.unknown()
                        )
                    code, ret_info = getattr(parent_handler, name)(*args, var_info=self.var_info, tile_desc=self.kernel_group.tile_desc)
                    csevar = self.cse.generate(
                        self.compute,
                        code,
                        bounds=buf_bounds,
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
                return sympy_symbol(str(index_var))  # type: ignore[attr-defined]

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
                return self.load(name, index)

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


