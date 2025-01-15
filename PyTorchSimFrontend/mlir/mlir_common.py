import os
import torch
from torch._inductor.codegen import common
from torch._inductor.codegen import cpp
from torch._inductor.virtualized import V
from torch._inductor.ir import MultiOutputLayout
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

    def mlir_argdefs(self, extra_node=dict()):
        buffer_types = {}
        for x in V.graph.buffers:
            if not isinstance(x.layout, MultiOutputLayout): # FIXME: MultiOutputLayout should be handled
                buffer_types[x.get_name()] = [x.get_dtype(), x.get_numel()]
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                buffer_types[name] = [get_sympy_Expr_dtype(val), 1]
            else:
                buffer_types[name] = [val.get_dtype(), val.get_numel()]
        buffer_types.update(
            {name: val.dtype for name, val in V.graph.constants.items()}
        )
        buffer_types.update(
            {name: [val.get_dtype(), val.get_numel()] for name, val in extra_node.items()}
        )

        call_args = []
        arg_defs = []
        arg_attributes = []
        def set_info(outer, inner, arg_type):
            arg_defs.append(f"%{inner}: memref<{buffer_types[outer][1]}x{DTYPE_TO_MLIR[buffer_types[outer][0]]}>")
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

    def get_tile_shape(self):
        return f"{self.n_row}x{self.n_col}"

    def get_chunk_size(self):
        if self.tile_layout == self.TILE_ROW_WISE:
            chunk_size = self.get_tile_size_per_lane()
        else:
            chunk_size = self.get_cols_per_lane()
        return chunk_size

    @staticmethod
    def div_round_up(size, round_val):
        return (size + round_val - 1) // round_val

class MLIRWrapperKenrelGroup(cpp.KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = MLIRKernelArgs()

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

    def __init__(self, args=None):
        super().__init__(args)
        self.kernel_group : MLIRWrapperKenrelGroup = None
        # Kernel iteration range info
        self.call_ranges = None
        self.ranges = None
        self.reduction_depth = None
        self.itervars = None
        # Code buffer
        self.vector_compute = IndentedBuffer()
        self.reductions_suffix = IndentedBuffer()
        self.cse = common.CSE(self.newvar_prefix, self.suffix)
        # Tile size setting
        tile_row = extension_config.CONFIG_TILE_ROW
        if tile_row == -1:
            tile_row = self.vlen * self.vector_lane
        tile_col = extension_config.CONFIG_TILE_COL
        if tile_col == -1:
            tile_col = 8 # FIXME: tile_col is not always vector_lane * vlen
        self.tile_desc = MLIRTile(tile_row, tile_col, self.vector_lane)
        self.var_info = {} # MLIR variable info
        self.buffer_types : dict = None
        self.read_writes = None

    def set_ranges(self, lengths, reduction_lengths, read_writes):
        self.read_writes = read_writes
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

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        self.set_ranges(group, reduction_group, None)
        with self as kernel:
            kernel.args = kernel.kernel_group.args
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group, node.read_writes)
                kernel.args.tile_row = kernel.tile_desc.n_row
                kernel.args.tile_col = kernel.tile_desc.n_col
                _, _, _, kernel.buffer_types = kernel.args.mlir_argdefs()
                node.run(vars, reduction_vars)
        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()
        return src_code

    def codegen_kernel(self, kernel_name):
        arg_defs, _, _, _ = self.kernel_group.args.mlir_argdefs()
        code = self._codegen_kernel(arg_defs, kernel_name)
        return code.getvalue()

    def _codegen_kernel(self, arg_defs, kernel_name):
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
        return code

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
            x: self.args.size(x)
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
                    code, ret_info = getattr(parent_handler, name)(*args, var_info=self.var_info)
                    csevar = self.cse.generate(
                        self.compute,
                        code,
                        bounds=buf_bounds,
                    )
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


