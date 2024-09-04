import torch
from torch._inductor.codegen import common
from torch._inductor.virtualized import V
import sympy

from typing import Callable

import sympy

import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges

from torch._inductor.utils import (
    free_symbol_startswith,
    get_sympy_Expr_dtype,
    IndentedBuffer,
    sympy_subs,
    unique,
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
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]

class MLIRKernelArgs(common.KernelArgs):
    MLIR_ARGS_IN = 0x01
    MLIR_ARGS_OUT = 0x02
    MLIR_ARGS_INOUT = 0x04
    MLIR_ARGS_VAR = 0x08

    @staticmethod
    def is_mlir_arg_in(value):
        return (MLIRKernelArgs.MLIR_ARGS_IN & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    @staticmethod
    def is_mlir_arg_out(value):
        return (MLIRKernelArgs.MLIR_ARGS_OUT & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    def mlir_argdefs(self, only_args=False):
        buffer_types = {x.get_name(): [x.get_dtype(), x.get_numel()] for x in V.graph.buffers}
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                buffer_types[name] = [get_sympy_Expr_dtype(val), 1]
            else:
                buffer_types[name] = [val.get_dtype(), val.get_numel()]
        buffer_types.update(
            {name: val.dtype for name, val in V.graph.constants.items()}
        )

        call_args = []
        arg_defs = []
        arg_attributes = {}
        def set_info(outer, inner, arg_type):
            arg_defs.append(f"%{inner}: memref<{buffer_types[outer][1]}x{DTYPE_TO_MLIR[buffer_types[outer][0]]}>")
            call_args.append(outer)
            arg_attributes[outer] = [arg_type] + buffer_types[outer]

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

class BaseMLIRKernel(common.Kernel):
    newvar_prefix = "%"
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, args=None):
        super().__init__(args)
        self.vector_compute = IndentedBuffer()
        self.reductions_suffix = IndentedBuffer()
        self.cse = common.CSE(self.newvar_prefix, self.suffix)
        # Default HW setting
        self.vector_lane = 4
        self.tile_size = 4
        self.tile_info = {}

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def store(self, name, index, value, mode=None):
        raise NotImplementedError()

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def expand(self, args, buf_bounds):
        if not args[0] in self.tile_info or not args[1] in self.tile_info:
            return args, 1, torch.float32 # FIXME: dtype is not always float32
        lhs_tile_size, lhs_dtype = self.tile_info[args[0]]
        rhs_tile_size, rhs_dtype = self.tile_info[args[1]]
        lhs_shape = f"vector<{lhs_tile_size}x{DTYPE_TO_MLIR[lhs_dtype]}>" if lhs_tile_size > 1 else DTYPE_TO_MLIR[lhs_dtype]
        rhs_shape = f"vector<{rhs_tile_size}x{DTYPE_TO_MLIR[rhs_dtype]}>" if rhs_tile_size > 1 else DTYPE_TO_MLIR[rhs_dtype]
        temp = list(args)
        if lhs_tile_size > rhs_tile_size:
            expand = f"vector.broadcast %{args[1]} : {rhs_shape} to {lhs_shape}"
            temp[1] = self.cse.generate(self.compute, expand, bounds=buf_bounds)
        elif lhs_tile_size < rhs_tile_size:
            expand = f"vector.broadcast %{args[0]} : {lhs_shape} to {rhs_shape}"
            temp[0] = self.cse.generate(self.compute, expand, bounds=buf_bounds)
        args = tuple(temp)
        return args, max(lhs_tile_size, rhs_tile_size), lhs_dtype

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

                    args, tile_size, dtype = self.expand(args, buf_bounds)
                    csevar = self.cse.generate(
                        self.compute,
                        getattr(parent_handler, name)(*args, tile_size=tile_size, **kwargs),  # type: ignore[has-type]
                        bounds=buf_bounds,
                    )
                    self.tile_info[csevar] = tile_size, dtype
                    csevar.update_on_args(name, args, kwargs)
                    return csevar

                return inner

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                # Skip CSE since this doesn't return an expression
                return self.indirect_indexing(index_var, size, check)  # type: ignore[attr-defined]

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
