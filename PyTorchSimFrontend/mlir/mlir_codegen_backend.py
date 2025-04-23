import contextlib
import sympy
import re
import os
import math
import torch
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.codecache import write_atomic, write
from torch._inductor.utils import (
    IndentedBuffer,
    is_welford_reduction,
)
from torch.utils._sympy.functions import ModularIndexing
import PyTorchSimFrontend.extension_codecache as extension_codecache

from PyTorchSimFrontend import extension_config
from . import mlir_common
from .mlir_common import LoopLevel, LoopNest

def reduction_init(reduction_type, dtype):
    if dtype in cpp.DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return float(0) if dtype.is_floating_point else int(0)
    if reduction_type == "prod":
        return float(1) if dtype.is_floating_point else int(1)
    if reduction_type in {"max", "argmax"}:
        if dtype == torch.float32:
            return f"0x{mlir_common.MLIR_INF['-inf']['f32']:x}"
        elif dtype == torch.float64:
            return f"0x{mlir_common.MLIR_INF['-inf']['f64']:x}"
        else:
            return "0.0"
    if reduction_type in {"min", "argmin"}:
        if dtype == torch.float32:
            return f"0x{mlir_common.MLIR_INF['inf']['f32']:x}"
        elif dtype == torch.float64:
            return f"0x{mlir_common.MLIR_INF['inf']['f64']:x}"
        else:
            return "0.0"
    if reduction_type in {"welford_reduce"}:
        return f"0.0"
    raise AssertionError(reduction_type)

def reduction_partial_combine_vec(reduction_type, vector_value, init_value):
    if reduction_type == "sum":
        return ops.add(vector_value, init_value)
    if reduction_type == "prod":
        return ops.mul(vector_value, init_value)
    if reduction_type == "max":
        return ops.maximum(vector_value, init_value)
    if reduction_type == "min":
        return ops.minimum(vector_value, init_value)
    if reduction_type == "any":
        return ops.logical_and(vector_value, init_value)
    raise AssertionError(reduction_type)

def reduction_combine_vec(reduction_type, vector_value, init_value, axis, shape, reduced_shape):
    if reduction_type == "sum":
        return f"vector.multi_reduction <add>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "prod":
        return f"vector.multi_reduction <mul>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "max":
        return f"vector.multi_reduction <maximumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "min":
        return f"vector.multi_reduction <minimumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "any":
        return f"vector.multi_reduction <and>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    raise AssertionError(reduction_type)

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty, empty_strided
                from {extension_codecache.__name__} import CustomAsyncCompile
                from PyTorchSimFrontend.extension_op import sparse_mm_dummy_stonne_outer
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.aten._reinterpret_tensor
                async_compile = CustomAsyncCompile()
                os.environ["TORCHSIM_LAST_COMPILED_MODULE"] = __file__
            """
        )

class ExtensionOverrides(common.OpOverrides):
    # Binary element wise operations
    @staticmethod
    def custom_cast(operand, target_type, *args, var_info=None, **kwargs):
        dtype = var_info[operand][1]
        if dtype == "index":
            ret = ops.index_cast(operand, target_type, var_info=var_info)
        else:
            ret = ops.to_dtype(operand, target_type, var_info=var_info)
        return ret, var_info[ret]

    @staticmethod
    def binary_elementwise_common(operand1, operand2, var_info):
        operand1.bounds = operand1.bounds.unknown()
        operand2.bounds = operand2.bounds.unknown()
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]
        # Tile size check
        if op_type1[0] != op_type2[0]:
            # Try to broad cast
            lhs_tile_size, lhs_dtype = op_type1
            rhs_tile_size, rhs_dtype = op_type2
            if lhs_tile_size > rhs_tile_size:
                operand2 = ops.broadcast(operand2, operand1, var_info=var_info)
                op_type2 = var_info[operand2]
            elif lhs_tile_size < rhs_tile_size:
                operand1 = ops.broadcast(operand1, operand2, var_info=var_info)
                op_type1 = var_info[operand1]

        # Data type check
        if op_type1[1] != op_type2[1]:
            if op_type1[1] == "index" or op_type1 == "index":
                if op_type1[1] == "index":
                    operand1 = ops.index_cast(operand1, op_type2[1], var_info)
                    op_type1 = var_info[operand1]
                if op_type2[1] == "index":
                    operand2 = ops.index_cast(operand2, op_type1[1], var_info)
                    op_type2 = var_info[operand2]
            elif op_type1[1][0] == "i" and op_type2[1][0] == "f":
                operand1 = ops.to_dtype(operand1, op_type2[1], var_info)
                op_type1 = var_info[operand1]
            elif op_type1[1][0] == "f" and op_type2[1][0] == "i":
                operand2 = ops.to_dtype(operand2, op_type1[1], var_info)
                op_type2 = var_info[operand2]
            elif op_type1[1][0] == op_type2[1][0]:
                if int(op_type1[1][1:]) > int(op_type2[1][1:]):
                   operand2 = ops.ext(operand2, op_type1[1])
                   op_type2 = var_info[operand2]
                elif int(op_type1[1][1:]) < int(op_type2[1][1:]):
                   operand1 = ops.ext(operand1, op_type2[1])
                   op_type1 = var_info[operand1]
            else:
                raise NotImplementedError("Unsupported type converting")

        # Updated var info
        tile_size = op_type1[0]
        ret_type = op_type1[1]
        return tile_size, ret_type, operand1, operand2

    @staticmethod
    def add(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.add{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def sub(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.sub{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def mul(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.mul{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def div(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.divf'
        else:
            opcode = f'arith.divui'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def truediv(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.divf'
        else:
            opcode = f'arith.divui'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def modular(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            raise NotImplementedError("Not support remainder operation for floating point")
        else:
            opcode = f'arith.remui'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def minimum(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.minimumf'
        else:
            opcode = f'arith.minimumui'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def maximum(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.maximumf'
        else:
            opcode = f'arith.maximumui'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def to_dtype(operand, dst_mlir_dtype, *args, var_info=None, **kwargs):
        src_mlir_dtype = var_info[operand][1]
        tile_size = var_info[operand][0]
        if isinstance(dst_mlir_dtype, torch.dtype):
            dst_mlir_dtype = mlir_common.DTYPE_TO_MLIR[dst_mlir_dtype]
        dst_bits = int(dst_mlir_dtype[1:])
        src_bits = int(src_mlir_dtype[1:])
        shape = f"vector<{tile_size}x{dst_mlir_dtype}>" if tile_size > 1 else dst_mlir_dtype
        src_shape = f"vector<{tile_size}x{src_mlir_dtype}>" if tile_size > 1 else src_mlir_dtype
        if dst_mlir_dtype[0] == "i" and src_mlir_dtype[0] == "f":
            return f"arith.fptoui%{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
        if dst_mlir_dtype[0] == "f" and src_mlir_dtype[0] == "i":
            return f"arith.uitofp%{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
        if dst_mlir_dtype[0] == "i":
            if dst_bits > src_bits:
                return f"arith.extui %{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
            elif dst_bits < src_bits:
                return f"arith.trunc %{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
            return f"arith.maximumi %{operand}, %{operand} : {shape}", [tile_size, dst_mlir_dtype]
        elif dst_mlir_dtype[0] == "f":
            if dst_bits > src_bits:
                return f"arith.extf %{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
            elif dst_bits < src_bits:
                return f"arith.trunf %{operand} : {src_shape} to {shape}", [tile_size, dst_mlir_dtype]
            return f"arith.maximumf %{operand}, %{operand} : {shape}", [tile_size, dst_mlir_dtype]
        else:
            raise NotImplementedError("Unsupported type for to_dtype ops")

    @staticmethod
    def constant(value, src_type, *args, var_info=None, **kwargs):
        if isinstance(src_type, torch.dtype):
            src_type = mlir_common.DTYPE_TO_MLIR[src_type]

        if "inf" == str(value) or "-inf" == str(value) or "nan" == str(value):
            value = f"0x{mlir_common.MLIR_INF[str(value)][src_type]:x}"
        # if value represented by e notation, convert to float (ex 1e-3 -> 1.0e-3)
        elif "e" in str(value):
            value = format(float(value), ".20f")
        elif src_type[0] == "f":
            value = format(value, ".20f")
        elif src_type[0] == "i":
            value = int(value)
        return f'arith.constant {value} : {src_type}', [1, src_type]

    @staticmethod
    def alloc(size, src_type, *args, var_info=None, **kwargs):
        return f"memref.alloc() : memref<{size}x{src_type}>", [size, src_type]

    @staticmethod
    def extractelement(operand, idx, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f"vector.extract %{operand}[{idx}]: {dtype} from {shape}", [1, dtype]

    # transcendental functions
    @staticmethod
    def exp(operand, *args, var_info=None, **kwargs):
        # Check scalar
        op_type = var_info[operand]
        if op_type[0] == 1:
            val = ops.constant(0, op_type[1])
            var_info[val][0] = 4
            operand = ops.broadcast(operand, val)
            val = ops.exp(operand)
            result = ops.extractelement(val, 0)
            return result, var_info[result]
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.exp %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def erf(operand, *args, var_info=None, **kwargs):
        # Check scalar
        op_type = var_info[operand]
        if op_type[0] == 1:
            val = ops.constant(0, op_type[1])
            var_info[val][0] = 4
            operand = ops.broadcast(operand, val)
            val = ops.exp(operand)
            result = ops.extractelement(val, 0)
            return result, var_info[result]
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.erf %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def tanh(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]

        # Check scalar
        op_type = var_info[operand]
        if op_type[0] == 1:
            val = ops.constant(0, op_type[1])
            var_info[val][0] = 4
            operand = ops.broadcast(operand, val)
            val = ops.exp(operand)
            result = ops.extractelement(val, 0)
            return result, var_info[result]
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.tanh %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def sqrt(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.sqrt %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def rsqrt(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.rsqrt %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def pow(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        # Type check & auto cast
        if ret_type[0] != "f":
            operand1, ret_type = ops.to_dtype(operand1, "f32", var_info=var_info)
            var_info[operand1] = ret_type

        # Type check & auto cast
        if ret_type[0] != "f":
            operand2, ret_type = ops.to_dtype(operand2, "f32", var_info=var_info)
            var_info[operand2] = ret_type

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f"math.pow{ret_type[0]} %{operand1}, %{operand2} : {shape}", [tile_size, ret_type]

    @staticmethod
    def log(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.log %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def reciprocal(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype

        return ops.div(ops.constant(1.0, dtype), operand), [tile_size, dtype]

    @staticmethod
    def ext(operand, dtype, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        shape = f"vector<{op_type[0]}x{op_type[1]}>" if op_type[0] > 1 else f"{op_type[1]}"
        target_type = f"vector<{op_type[0]}x{dtype}>" if op_type[0] > 1 else f"{dtype}"
        if op_type[0] == "f":
            opcode = f'arith.extf'
        else:
            opcode = f'arith.extui'
        return f'{opcode} %{operand} : {shape} to {target_type}', [op_type[0], dtype]

    # Logical operations
    @staticmethod
    def neg(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype[0] != "f":
            operand, dtype = ops.to_dtype(operand, "f32", var_info=var_info)
            var_info[operand] = dtype

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.negf %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def eq(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "oeq"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "eq"
        else:
            raise ValueError(f"Unsupported data type for 'eq' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def ne(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "one"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sne"
        else:
            raise ValueError(f"Unsupported data type for 'ne' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def lt(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "olt"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "slt"
        else:
            raise ValueError(f"Unsupported data type for 'lt' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def gt(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "ogt"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sgt"
        else:
            raise ValueError(f"Unsupported data type for 'gt' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def le(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "ole"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sle"
        else:
            raise ValueError(f"Unsupported data type for 'le' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def ge(operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "oge"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sge"
        else:
            raise ValueError(f"Unsupported data type for 'ne' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def and_(operand1, operand2, *args, var_info=None, **kwargs):
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]

        # Type check & auto cast
        if op_type1[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand1] = dtype

        # Type check & auto cast
        if op_type2[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand2] = dtype

        ret_type = op_type1[1]
        tile_size = op_type1[0]

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.andi %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def or_(operand1, operand2, *args, var_info=None, **kwargs):
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]

        # Type check & auto cast
        if op_type1[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand1] = dtype

        # Type check & auto cast
        if op_type2[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand2] = dtype

        ret_type = op_type1[1]
        tile_size = op_type1[0]

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.ori %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def xor(operand1, operand2, *args, var_info=None, **kwargs):
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]

        # Type check & auto cast
        if op_type1[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand1] = dtype

        # Type check & auto cast
        if op_type2[1][0] != "i":
            operand1, dtype = ops.to_dtype(operand1, "i32", var_info=var_info)
            var_info[operand2] = dtype

        ret_type = op_type1[1]
        tile_size = op_type1[0]

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.xori %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]


    @staticmethod
    def logical_and(operand1, operand2, *args, var_info=None, **kwargs):
        op_type = var_info[operand1]
        # Type check & auto cast
        if op_type[1] != "i1":
            raise NotImplementedError("Logical operation with not bool data type")
        return ExtensionOverrides.and_(operand1, operand2, *args, var_info=var_info, **kwargs)

    @staticmethod
    def logical_not(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]

        ret_type = op_type[1]
        tile_size = op_type[0]
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        const_one = ops.constant(0, ret_type)
        const_one = ops.broadcast(const_one, operand, var_info=var_info)
        ret = ops.eq(operand,const_one)
        return ret, [tile_size, var_info[ret]]

    @staticmethod
    def logical_or(operand1, operand2, *args, var_info=None, **kwargs):
        op_type = var_info[operand1]
        # Type check & auto cast
        if op_type[1] != "i1":
            raise NotImplementedError("Logical operation with not bool data type")
        return ExtensionOverrides.or_(operand1, operand2, *args, var_info=var_info, **kwargs)

    @staticmethod
    def logical_xor(operand1, operand2, *args, var_info=None, **kwargs):
        op_type = var_info[operand1]
        # Type check & auto cast
        if op_type[1] != "i1":
            raise NotImplementedError("Logical operation with not bool data type")
        return ExtensionOverrides.xor(operand1, operand2, *args, var_info=var_info, **kwargs)

    @staticmethod
    def relu(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        ret_type = "f32"
        return ops.maximum(operand, ops.constant(0.0, "f32")), [tile_size, ret_type]

    @staticmethod
    def sigmoid(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        ret_type = "f32"
        one = ops.constant(1, "f32")
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(operand)))), [tile_size, ret_type]

    # Special operaitons
    @staticmethod
    def where(condition, operand1, operand2, *args, var_info=None, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2, var_info)
        cond_type = var_info[condition]
        if cond_type[0] < tile_size:
            condition = ops.broadcast(condition, operand1, var_info=var_info)
        elif cond_type[0] > tile_size:
            operand1 = ops.broadcast(operand1, condition, var_info=var_info)
            operand2 = ops.broadcast(operand2, condition, var_info=var_info)
        tile_size, ret_type = var_info[operand1]

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        cond_shape = f"vector<{tile_size}xi1>," if tile_size > 1 else ""
        return f"arith.select %{condition}, %{operand1}, %{operand2} : {cond_shape} {shape}", [tile_size, ret_type]


    @staticmethod
    def masked(mask, body, other, *args, var_info=None, tile_size=16, dtype="f32", ninf_declared=False, **kwargs):
        result = body()
        val = ops.constant(other, dtype, *args, **kwargs)
        result = ops.where(mask, result, val)
        return result, var_info[result]

    @staticmethod
    def index_cast(operand, target_type, *args, var_info=None, **kwrags):
        op_type = var_info[operand]
        src_shape = f"vector<{op_type[0]}x{op_type[1]}>" if op_type[0] > 1 else op_type[1]
        des_shape = f"vector<{op_type[0]}x{target_type}>" if op_type[0] > 1 else target_type
        return f"arith.index_cast %{operand} : {src_shape} to {des_shape}", [op_type[0], target_type]

    @staticmethod
    def broadcast_unflat(operand1, operand2, *args, var_info=None, **kwargs):
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]
        src_shape = f"vector<{op_type1[0]}x{op_type1[1]}>"# if op_type1[0] > 1 else op_type1[1]
        des_shape = f"vector<{op_type2[0]//op_type1[0]}x{op_type1[0]}x{op_type1[1]}>"# if op_type2[0] > 1 else op_type1[1] # Use tile size only

        expand = f"vector.broadcast %{operand1} : {src_shape} to {des_shape}"
        return expand, [op_type2[0], op_type1[1]]

    @staticmethod
    def broadcast(operand1, operand2, *args, var_info=None, **kwargs):
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]
        src_shape = f"vector<{op_type1[0]}x{op_type1[1]}>" if op_type1[0] > 1 else op_type1[1]
        des_shape = f"vector<{op_type2[0]}x{op_type1[1]}>" # if op_type2[0] > 1 else op_type1[1] # Use tile size only

        # Special case for length 2 vector. We used this vector to avoid scalar operations...
        if op_type1[0] != 1 and op_type2[0] % op_type1[0] == 0:
            unflat_operand = ops.broadcast_unflat(operand1, operand2)
            unflat_shape = f"vector<{op_type2[0]//op_type1[0]}x{op_type1[0]}x{op_type1[1]}>"
            expand = f"vector.shape_cast %{unflat_operand} : {unflat_shape} to {des_shape}"
        elif op_type1[0] == 1:
            expand = f"vector.broadcast %{operand1} : {src_shape} to {des_shape}"
        else:
            raise NotImplementedError("Not supporting broadcast type...")
        return expand, [op_type2[0], op_type1[1]]

RTYPE_TO_MLIR = {
    "sum": "add",
    "prod": "mul",
}

DMA_TYPE = {
    "MVIN1": 2,
    "MVIN2": 1,
    "MVIN3": 14,
    "MVOUT1": 3,
}

class MLIRKernel(mlir_common.BaseMLIRKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, kernel_group):
        super().__init__(kernel_group)
        self.const_buffer = IndentedBuffer()
        self.alloc_buffer = IndentedBuffer()
        self.spad_buffer = IndentedBuffer()
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.applys = IndentedBuffer()
        self.body = IndentedBuffer()
        self.dma_loads = IndentedBuffer()
        self.dma_stores = IndentedBuffer()
        self.indexed_buffer = IndentedBuffer()
        self.global_vars = IndentedBuffer()
        self.header = IndentedBuffer()
        self.gem5_header = IndentedBuffer()
        self.header.writeline("#include <unistd.h>")
        self.header.writeline("#include <stdlib.h>")
        self.header.writeline("void* __wrap_malloc(size_t size) { return sbrk(size); }")
        self.header.writeline("void __wrap_free(void *ptr) { return; }")
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.spad_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="spad")
        self.apply_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="apply")
        self.iterator_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="iter")
        self.init_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init")
        self.init_vec_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init_vec")
        self.const_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="const")
        self.alloc_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="alloc")
        self.indexed_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="indexed_op")
        self.map_cse = common.CSE("#", self.suffix, name_prefix="map")
        self.global_vars_dict = dict()
        self.reduction_vars = dict()
        self.consts = dict()
        self.tags = dict()
        self.dma_read_cache = dict()
        self.dma_write_cache = dict()
        self.dma_read_counter = 1
        self.dma_write_counter = 1
        self.affine_yield = {}
        self.welford_reduce_out = None
        self.reduce_iterator = {}
        self.is_template_kernel = False
        self.spad_buffer_dict = dict()
        self.base_vector_initialized = False

    def reset(self):
        self.__init__(self.kernel_group)

    # padding type 0: zero-padding 1: negative-padding(-inf) ...
    def get_padding_type(self):
        ops = self.current_node.node.origins
        if self.current_node.is_reduction():
            for op in ops:
                if "exp" in op.name: # exponential reduciton case
                    return 1
        # for op in ops: # TODO: padding has some problem in the case of max_pool
        #     if "max_pool" in op.args[0].name:
        #         return 1
        return 0

    def convert_index(self, expr, buffer):
        if len(expr.free_symbols) != 1:
            raise NotImplementedError("Not supporting this view operation...!")

        if expr.is_symbol:
            return expr

        expr_str = str(expr)
        if isinstance(expr, ModularIndexing):
            replace_str = f"({expr.args[0]} floordiv {expr.args[1]}) mod {expr.args[2]}"
            expr_str = re.sub(r"ModularIndexing\([^)]*\)", replace_str, expr_str)
        elif "//" in expr_str:
            expr_str = expr_str.replace("//", " floordiv ")
        else:
            raise NotImplementedError("What is this case?")

        indices = [expr.args[0]]
        args = ", ".join(map(str, indices))
        map_var = self.map_cse.generate(self.global_vars, f"affine_map<({args}) -> ({expr_str})>")
        args = ", ".join([f"%{i}" for i in indices])
        index = self.apply_cse.generate(buffer, f"affine.apply #{map_var}({args})")
        return index

    def parse_indices(self, expr, buffer=None) -> common.CSEVariable:
        if buffer is None:
            buffer = self.applys

        # Constant case
        if expr.is_number:
            return self.get_const_cse(int(expr))

        # Identity case
        if len(expr.args) == 0:
            return expr

        args = list(expr.args)
        # Sort index variable.. ex) (%index1, %index0)
        args_dict = {term: list(term.free_symbols)[0] for term in args if term.free_symbols}
        sorted_args = sorted(args_dict.keys(), key=lambda term: str(args_dict[term]))
        indices = []
        for arg in sorted_args:
            if arg.is_Mul and arg.args[0].is_number:
                new_arg = sympy.Symbol(str(self.convert_index(arg.args[1], buffer)))
                expr = expr.replace(arg.args[1], new_arg)
                indices.append(str(new_arg))
            elif not arg.is_number:
                new_arg = sympy.Symbol(str(self.convert_index(arg, buffer)))
                expr = expr.replace(arg, new_arg)
                indices.append(str(new_arg))

        # Extract index var
        expr_str = str(expr)
        args = ", ".join(map(str, indices))
        map_var = self.map_cse.generate(self.global_vars, f"affine_map<({args})[] -> ({expr_str})>")
        args = ", ".join([f"%{i}" for i in indices])
        index = self.apply_cse.generate(buffer, f"affine.apply #{map_var}({args})[]")
        return index

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        index = self.convert_indirect_indexing(index)
        padding = self.get_padding_type()
        dram_var = self.kernel_group.args.input(name)

        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        local_tile_desc, index_var = self.get_dma_info(name, index)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        # Define scratch pad buffer
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)

        # MVIN Encoding
        code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride, padding)
        self.cse.generate(self.dma_loads, code, assignment = False) # FIXME: assignment = False does not support caching
        compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])
        # Generate vector load instruction
        if compute_vec_size > 1:
            operation = "affine.vector_load"
            line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.load"
            line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [compute_vec_size, mlir_dtype])
        self.spad_buffer_dict[str(out)] = [sram_var, local_tile_desc.get_tile_size(), tile_numel_per_lane, sram_index_var, tile_shape, vshape]
        return out

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        # Prepare dma instruction
        local_tile_desc, index_var = self.get_dma_info(name, index)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()
        tile_size = local_tile_desc.get_tile_size()

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()
        require_store = True

        if str(value) in self.spad_buffer_dict:
            # Todo. If tile_size is not same (i.e., view operation), we can't apply peephole optimization easily
            require_store = self.spad_buffer_dict[str(value)][1] != tile_size

        if require_store:
            # Define scratch pad buffer
            sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var, index)
            compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])
            # Generate vector store instruction
            store_size, operand_type = self.var_info[value]
            if mlir_dtype != operand_type:
                value = ops.custom_cast(value, mlir_dtype, var_info=self.var_info)

            if compute_vec_size > 1 and store_size > 1:
                operation = "affine.vector_store"
                line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
            else:
                operation = "affine.store"
                line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}"
            self.stores.writeline(common.DeferredLine(name, line)) # TODO: Should be changed to self.compute?
        else:
            sram_var = self.spad_buffer_dict[str(value)][0]
            sram_index_var = self.spad_buffer_dict[str(value)][3]

        # Generate DMA instruction
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.dma_stores.writeline(common.DeferredLine(name, code))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        elif is_welford_reduction(reduction_type):
            if reduction_type == "welford_combine":
                raise NotImplementedError("welford_combine")
            else:
                assert reduction_type == "welford_reduce"
                type_name = mlir_common.DTYPE_TO_MLIR[dtype]
                reduction_key = src_dtype, reduction_type, value
                sum = self.reduction(dtype, src_dtype, "sum", value)
                sqr_sum = self.reduction(dtype, src_dtype, "sum", ops.mul(value, value))
                self.welford_reduce_out = (sum, sqr_sum, None)
                return sum, sqr_sum, None

        # Prepare reduction loop
        reduction_key = src_dtype, reduction_type, value
        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        iterator = self.iterator_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        init = self.init_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        init_vec = self.init_vec_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        init = self.cse.generate(self.reduction_prefix, f"arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
        vec_len = self.kernel_group.tile_desc.get_compute_vec_size()
        reduced_shape = self.kernel_group.tile_desc.get_mlir_vshape(type_name)

        # Set accumulation var
        if vec_len == 1: # 1-D vector to scalar
            # Edge case for scalar
            init_vec = init
        else:
            # Adjust shape and inital value
            init_vec = self.cse.generate(self.reduction_prefix, f"vector.broadcast %{init} : {type_name} to {reduced_shape}")
        acc_var = init_vec

        # Reduction body prepare
        body_acc = self.reduction_cse.generate(
            self.compute, f"reduction {reduction_key}body_acc", write=False
        )
        body_iter_arg = self.iterator_cse.generate(
            self.compute, f"reduction {reduction_key}body_iter_arg", write=False
        )
        self.register_var_info(body_iter_arg, [vec_len, type_name])

        self.reduction_vars[acc] = (reduction_type, iterator, acc_var, reduced_shape)
        self.affine_yield[body_acc] = reduced_shape
        self.reduction_cse.reduction_cache[reduction_key] = acc
        self.iterator_cse.reduction_cache[reduction_key] = iterator
        self.init_cse.reduction_cache[reduction_key] = init_vec

        # Reduction body codegen
        result = reduction_partial_combine_vec(reduction_type, value, body_iter_arg)
        self.compute_body_loop.reduction_vars[body_acc] = (reduction_type, body_iter_arg, iterator, reduced_shape)
        self.compute_body_loop.affine_yield[result] = reduced_shape

        # Final reduction
        reduction_size = self.kernel_group.tile_desc.get_numel_per_lane() // self.kernel_group.tile_desc.get_tile_size()[-1]
        assert(vec_len % reduction_size==0)
        if vec_len > reduction_size:
            init = self.cse.generate(self.reductions_suffix, f"arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
            if reduction_size == 1:
                final_reduced_shape = f"{type_name}"
                out = self.cse.generate(self.reductions_suffix, reduction_combine_vec(reduction_type, acc, init, axis=0, shape=reduced_shape, reduced_shape=final_reduced_shape))
            else:
                final_reduced_shape = f"vector<{reduction_size}x{type_name}>"
                init_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %{init} : {type_name} to {final_reduced_shape}")
                new_vshape= f"vector<{vec_len//reduction_size}x{reduction_size}x{type_name}>"
                value = self.cse.generate(self.reductions_suffix, f"vector.shape_cast %{acc} : {reduced_shape} to {new_vshape}")
                out = self.cse.generate(self.reductions_suffix, reduction_combine_vec(reduction_type, value, init_vec, axis=0, shape=new_vshape, reduced_shape=final_reduced_shape))
            acc = out

        # reigster reduction output
        var_info = [reduction_size, mlir_common.DTYPE_TO_MLIR[dtype]]
        self.register_var_info(acc, var_info)
        return acc

    def store_reduction(self, name, index, value):
        # Note: Change cse temporaily
        # Store reduction can't share cached value stored in cse,
        # since it is not innermost loop body.
        tmp_cse = self.cse
        self.cse = self.reduction_cse

        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        index = self.rename_indexing(index)

        # Tile is always reuduced in inner loop
        local_tile_desc, index_var = self.get_dma_info(name, index, broadcast=False, store_reduction=True, buffer=self.reductions_suffix)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()
        compute_vec_size = self.kernel_group.tile_desc.get_numel_per_lane() // self.kernel_group.tile_desc.get_tile_size()[-1]
        if compute_vec_size == 1:
            vshape = f"{mlir_dtype}"
        else:
            vshape = f"vector<{compute_vec_size}x{mlir_dtype}>"
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, index_var,
                                                                         index, buffer=self.reduction_suffix)
        if self.welford_reduce_out is not None:
            # raise NotImplementedError()
            sum, sqr_sum, _ = self.welford_reduce_out
            # mean
            divider = self.cse.generate(self.reductions_suffix, f"arith.constant {float(self.ranges[self.reduction_depth])} : f32")
            if self.buffer_types[name][1] > 1:
                divider_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %{divider} : f32 to vector<{self.var_info[sum][0]}x{mlir_dtype}>")
            else:
                divider_vec = divider
            mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sum}, %{divider_vec} : {vshape}")

            # m2 = (E(X^2) - E(X)^2) * N
            sqr_mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sqr_sum}, %{divider_vec} : {vshape}")
            mean_sqr = self.cse.generate(self.reductions_suffix, f"arith.mulf %{mean}, %{mean} : {vshape}")
            variance = self.cse.generate(self.reductions_suffix, f"arith.subf %{sqr_mean}, %{mean_sqr} : {vshape}")
            m2 = self.cse.generate(self.reductions_suffix, f"arith.mulf %{variance}, %{divider_vec} : {vshape}")
            if self.current_node.node.origin_node: # FIXME: This is a temporary solution
                value = mean
            else:
                value = m2

        # Select src type
        if compute_vec_size == 1:
            operation = "affine.store"
            line = f"{operation} %{value}, %{sram_var}[{sram_index_var}] : {tile_shape}"
        else:
            operation =  "affine.vector_store"
            line = f"{operation} %{value}, %{sram_var}[{sram_index_var}] : {tile_shape}, {vshape}"
        self.reductions_suffix.writeline(common.DeferredLine(name, line))

        # MVOUT Encoding
        # Generate DMA instruction
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.reductions_suffix.writeline(common.DeferredLine(name, code))

        # Restore origin cse
        self.cse = tmp_cse

    def indirect_indexing(self, index_var, size, check=True):
        return str(index_var)

    def _index_expr(self, tile_size, renamed_expression, index, base_vector_index):
        tile_desc = self.kernel_group.tile_desc
        compute_vec_size = tile_desc.get_compute_vec_size()

        strides = [1] * len(tile_size)
        for i in range(len(tile_size) - 2, -1, -1):
            strides[i] = strides[i + 1] * tile_size[i + 1]

        # Create vector index
        compute_vec = self.cse.generate(self.compute, f"vector.broadcast %{self.compute_idx} : index to vector<{compute_vec_size}xindex>")
        self.register_var_info(compute_vec, [compute_vec_size, "index"])
        vector_index = ops.add(base_vector_index, compute_vec)

        # Create tile_dim index
        dim_list = []
        for idx in range(len(tile_size)):
            div_coeff = self.get_const_cse(strides[idx], "index")
            mod_coeff = self.get_const_cse(tile_size[idx], "index")
            div_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{div_coeff} : index to vector<{compute_vec_size}xindex>")
            mod_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{mod_coeff} : index to vector<{compute_vec_size}xindex>")
            self.register_var_info(div_vec, [compute_vec_size, "index"])
            self.register_var_info(mod_vec, [compute_vec_size, "index"])
            dim = ops.modular(ops.div(vector_index, div_vec), mod_vec)
            if idx == tile_desc.vlane_split_axis: # Need to add vector lane offset
                offset = tile_desc.vlane_stride * strides[idx]
                vlane_coeff = self.get_const_cse(0, "i64")
                vlane_vec_size = 4
                vlane_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{vlane_coeff} : i64 to vector<{vlane_vec_size}xi64>")
                vlane_offset = self.const_cse.generate(self.const_buffer, f"arith.addi %{vlane_vec}, %{vlane_vec} {{ vlane_offset={offset} }} : vector<{vlane_vec_size}xi64> // vlane offset")
                self.register_var_info(vlane_offset, [vlane_vec_size, "i64"])
                vlane_offset = ops.index_cast(vlane_offset, "index")
                self.register_var_info(vlane_offset, [vlane_vec_size, "index"])
                dim = ops.add(dim, vlane_offset)
            dim_list.append(dim)

        indices = [str(i) for i in index.free_symbols]
        for idx in indices:
            i = int(idx[5:])
            index_vec = self.cse.generate(self.compute, f"vector.broadcast %{idx} : index to vector<{compute_vec_size}xindex>")
            self.register_var_info(index_vec, [compute_vec_size, "index"])
            offset = ops.add(index_vec, dim_list[i])
            dim_list[i] = offset
        arg_lists = []
        for arg in renamed_expression.args:
            if isinstance(arg, sympy.Integer):
                offset = self.get_const_cse(int(arg))
                offset_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{offset} : index to vector<{compute_vec_size}xindex>")
                self.register_var_info(offset_vec, [compute_vec_size, "index"])
                arg_lists.append(offset_vec)
            elif isinstance(arg, sympy.Mul):
                if isinstance(arg.args[0], sympy.Integer) and isinstance(arg.args[1], sympy.Symbol):
                    coeff = self.get_const_cse(int(arg.args[0]))
                    coeff_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{coeff} : index to vector<{compute_vec_size}xindex>")
                    self.register_var_info(coeff_vec, [compute_vec_size, "index"])
                    result = ops.mul(dim_list[int(str(arg.args[1])[1:])], coeff_vec)
                    arg_lists.append(result)
                elif isinstance(arg.args[1], sympy.Integer) and isinstance(arg.args[0], sympy.Symbol):
                    coeff = self.get_const_cse(int(arg.args[1]))
                    coeff_vec = self.cse.generate(self.compute, f"vector.broadcast %{coeff} : index to vector<{compute_vec_size}xindex>")
                    self.register_var_info(coeff_vec, [compute_vec_size, "index"])
                    result = ops.mul(dim_list[int(str(arg.args[0])[1:])], coeff_vec)
                    arg_lists.append(result)
                else:
                    raise NotImplementedError("Not supporting format")
            elif isinstance(arg, sympy.Symbol):
                arg_lists.append(dim_list[int(str(arg)[1:])])
            else:
                raise NotImplementedError("Not supporting format")
        accum = arg_lists[0]
        for arg in arg_lists[1:]:
            accum = ops.add(accum, arg)
        return accum

    def index_expr(self, index, dtype):
        tile_desc = self.kernel_group.tile_desc
        tile_size = tile_desc.get_tile_size_per_lane()
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        str_tile_size = [str(dim) for dim in tile_size]
        compute_vec_size = tile_desc.get_compute_vec_size()
        tile_shape = f"memref<{compute_vec_size}xi64, 1>"
        vshape = f"vector<{compute_vec_size}xi64>"

        # Create base_vector index var
        c_type = "uint64_t"
        new_name = f"index_expr_{compute_vec_size}"
        if new_name not in self.global_vars_dict:
            self.header.writeline(f"{c_type} {new_name}_spad[{compute_vec_size}] __attribute__ ((section(\".spad\")));")
            self.gem5_header.writeline(f"{c_type} {new_name}_spad[{compute_vec_size}] __attribute__((aligned(64)));")
            self.global_vars.writeline(f"memref.global @{new_name}_spad : {tile_shape}")
            self.global_vars_dict[new_name] = []
        sram_var = self.spad_cse.generate(self.spad_buffer, f"memref.get_global @{new_name}_spad : {tile_shape}")
        # Initialize base vector
        if not self.base_vector_initialized:
            init_iter = "iter"
            parallel_map = f"affine.parallel (%{init_iter}) = ({0}) to ({compute_vec_size}) {{ // Base vector initializer"
            self.spad_buffer.writeline(parallel_map)
            with self.spad_buffer.indent():
                self.spad_buffer.writeline(f"%init_vec = vector.broadcast %{init_iter} : index to vector<2xindex>")
                self.spad_buffer.writeline(f"%init_cvt_vec = arith.index_cast %init_vec : vector<2xindex> to vector<2xi64>")
                self.spad_buffer.writeline(f"affine.vector_store %init_cvt_vec, %{sram_var}[%{init_iter}] : {tile_shape}, vector<2xi64>")
            self.spad_buffer.writeline("}")
            self.base_vector_initialized = True

        line = f"affine.vector_load %{sram_var}[0] : {tile_shape}, {vshape}"
        out = self.cse.generate(self.compute, line)
        self.register_var_info(out, [compute_vec_size, "i64"])
        base_vector_index = ops.index_cast(out, "index")
        self.register_var_info(base_vector_index, [compute_vec_size, "index"])

        renamed_symbols = {symbol: "d"+str(symbol)[5:] for symbol in index.free_symbols}
        renamed_expression = index.subs(renamed_symbols)
        result = self._index_expr(tile_size, renamed_expression, index, base_vector_index)
        return result

    def codegen_global_init(self):
        return self.global_vars

    def codegen_loops(self):
        code = mlir_common.ParallelLoopBuffer()
        # Loop body part
        tile_size = self.kernel_group.tile_desc.get_tile_size()
        # Apply paddings
        loops = [LoopLevel(var, size, step=step) for idx, (var, size, step) in enumerate(zip(self.itervars, self.ranges, tile_size))]
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars, self.affine_yield)
        # For non-loop code
        if (self.reduction_depth==0):
            loops = LoopNest([LoopLevel("dummy", 1)])

        if len(reductions.loops) > 1:
            NotImplementedError("Not support multiple reduction axis..")

        code.splice(self.const_buffer)
        code.splice(self.alloc_buffer)
        code.splice(self.spad_buffer)
        # Outerloop
        with contextlib.ExitStack() as stack:
            for loop in loops.loops:
                loop_lines = loop.lines()
                code.writelines(loop_lines)
                stack.enter_context(code.indent(attribute="{outer_loop=true}"))
            # Non-outerloop start
            code.splice(self.reduction_prefix)
            with contextlib.ExitStack() as stack:
                # Add reduction loops
                if len(reductions.loops):
                    reduction_lines = reductions.loops[0].lines()
                    epilogue = reductions.loops[0].epilogue_line()
                    code.writelines(reduction_lines)
                    stack.enter_context(code.indent(attribute="{accumulation_loop=true}", suffix=epilogue))
                code.splice(self.applys)
                code.splice(self.indexed_buffer)
                code.splice(self.dma_loads)
                # Compute body
                code.writelines(self.compute_body_loop.lines())
                with contextlib.ExitStack() as stack:
                    stack.enter_context(code.indent(attribute="{inner_loop=false}",suffix=self.compute_body_loop.epilogue_line()))
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.dma_stores)
            code.splice(self.reductions_suffix)
            # Non-outerloop end
        code.writeline(f"return")
        return code

    def codegen_nodes(self, nodes, kernel_name):
        src_code = ""

        n_try = 0
        while n_try < extension_config.CONFIG_MAX_AUTOTUNE_TRY:
            src_code = super().codegen_nodes(nodes, kernel_name)

            # Create extra headers for simulators
            write_path = extension_codecache.get_write_path(src_code)
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            spike_write_path = os.path.join(write_path, "global_var.h")
            gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
            if not os.path.exists(spike_write_path):
                spad_end_symbol = f"int spad_end[0] __attribute__ ((section(\".spad\"), aligned({self.spad_info['spad_size']*self.vector_lane})));"
                write_atomic(spike_write_path, self.header.getvalue() + spad_end_symbol)
            if not os.path.exists(gem5_write_path):
                write_atomic(gem5_write_path, self.gem5_header.getvalue())

            if not extension_config.CONFIG_AUTOTUNE:
                break

            try:
                bench_runner = self.run_bench(nodes, kernel_name, src_code)
                bench_runner(validate=True)
                print("Benchmark succeeded.")
                break
            except RuntimeError as e:
                if str(e) == "STACK_OVERFLOW":
                    n_try += 1
                    print(f"Benchmark failed due to stack overflow with tile size: {self.kernel_group.tile_desc.get_tile_size()}")
                    self.reset()
                else:
                    print(f"Benchmark failed with error: {str(e)}")
                    # raise e
        return src_code

    def get_dma_info(self, name, index, broadcast=True, store_reduction=False, buffer=None): # Need more argument?
        """
        A tile descriptor exists that is configured on a kernel group
        DMA desc should be adjusted according to buffer.
        Therefore, this function shoulde determin DRAM, SRAM stride and
        vectorlane mapping policy
        """
        # Use loads as default
        if buffer is None:
            buffer = self.applys if "tmp" not in str(index) else self.dma_loads

        # TODO.
        kg_tile_desc = self.kernel_group.tile_desc
        buffer_info = self.buffer_types[name]
        # Note: index could contain symbols that represent dynamic axies
        # Extract dimension of index(e.g, index0, index1)
        local_dims = [int(str(i)[5:]) for i in index.free_symbols if "index" in str(i)]
        implicit_local_dims = list(index.args)
        total_dims =  [int(str(i)[5:]) for i in self.itervars]
        local_tile_desc = mlir_common.MLIRMultiDimTile([1], self.vector_lane)
        local_dims.sort() # Assume that smaller index is placed in the outer loop
        indirect_dims = [f"{i}" for i in index.free_symbols if "tmp" in str(i)]
        indirect_arg_dims = [f"%{i}" for i in index.free_symbols if "tmp" in str(i)]
        for indirect_dim in indirect_dims:
            index = index.replace(sympy.Symbol(indirect_dim), 0)

        # Reduction can have two type of tile size
        if broadcast and (total_dims != local_dims or (self.reduction_depth!=len(total_dims) and total_dims[:self.reduction_depth] == local_dims)):
            # We have to create custom apply map to provide dram stride
            # ex) (d0, d1, ... dn, dn+1, dn+2, dk) -> (s0*d0 + s1*d1 + ... dn*0+ dn+1*0 + ... dk*0 + const)
            fake_dim = self.get_const_cse(0)
            input_expr = ",".join(["d"+str(i) for i in total_dims])
            output_expr = str(index).replace('index', 'd')
            input_argument = ",".join(["%index" + str(i) if i in local_dims else f"%{fake_dim}" for i in total_dims])
            map_var = self.map_cse.generate(self.global_vars, f"affine_map<({input_expr})[{','.join(indirect_dims)}] -> ({output_expr})>")
            index_var = self.apply_cse.generate(buffer, f"affine.apply #{map_var}({input_argument})[{','.join(indirect_arg_dims)}]")
            local_dims = total_dims # Brodatcast tile shape
        else:
            index_var = self.parse_indices(index, buffer=buffer)

        if kg_tile_desc.vlane_split_axis in local_dims:
            local_vlane_split_axis = local_dims.index(kg_tile_desc.vlane_split_axis)
        else:
            local_vlane_split_axis = max(len(local_dims) - 1, 0)

        # Case 0. Tile is 0-D scalar
        if len(local_dims) == 0:
            local_tile_desc.set_tile_size([kg_tile_desc.get_used_vlane() * kg_tile_desc.vlane_stride])         # Force it to use vector instruction.
            local_tile_desc.vlane_split_axis = local_vlane_split_axis    # last axis
            local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        # Case 1. Tile is 1-D vector type
        elif len(local_dims) == 1 and len(local_dims) <= self.reduction_depth:
            local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(local_dims[0])])
            local_tile_desc.vlane_split_axis = local_vlane_split_axis
            local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        # Case 2. Tile is 1-D vector type with reduction
        elif len(local_dims) == 1 and len(local_dims) == self.reduction_depth + 1:
            local_tile_desc.set_tile_size([1, kg_tile_desc.get_dim_size(local_dims[0])])
            local_tile_desc.vlane_split_axis = local_vlane_split_axis
            local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        # Case 3. Tile is 2-D tile
        elif len(local_dims) == 2:
            is_reduction = self.reduction_depth == 1 and not store_reduction
            if is_reduction:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims], [1, 0])
                local_tile_desc.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
            else:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
                local_tile_desc.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        # Case 3. Tile is 3-D tile
        elif len(local_dims) == 3:
            is_reduction = self.reduction_depth < 3 and not store_reduction
            if is_reduction:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims], [1, 2, 0])
                local_tile_desc.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
            else:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
                local_tile_desc.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        else:
            raise NotImplementedError("Currently not implemented... ;)")

        if len(implicit_local_dims)!=0 and len(local_dims) != len(implicit_local_dims) and self.is_modular_indexing(index):
            tile_size = local_tile_desc.get_tile_size()
            new_tile_size = []
            new_vlane_split_axis = local_tile_desc.vlane_split_axis
            implicit_dim_size = list(kg_tile_desc.implicit_dim_size.values())
            for i, target_dim_size in enumerate(implicit_dim_size):
                new_tile_size += [1]*(len(target_dim_size)-1) + tile_size[i:i+1]
                if local_tile_desc.vlane_split_axis >= i:
                    new_vlane_split_axis += len(target_dim_size)-1
            # Update
            local_tile_desc.set_tile_size(new_tile_size)
            local_tile_desc.vlane_split_axis = new_vlane_split_axis

        return local_tile_desc, index_var

    def get_dma_code(self, dma_type_name, attribute1, attribute2, mlir_dtype, dram_var, dram_index_var, sram_var, sram_index_var,
                     tag_name, dram_shape, tile_shape, tile_stride, padding_type=0):
        dma_key = (attribute1, attribute2, mlir_dtype)
        if dma_type_name == "MVIN" and dma_key in self.dma_read_cache:
            dma_type, attribute1, attribute2 = self.dma_read_cache[dma_key]
        elif dma_type_name == "MVOUT" and dma_key in self.dma_write_cache:
            dma_type, attribute1, attribute2 = self.dma_write_cache[dma_key]
        else:
            attribute1 = self.get_const_cse(attribute1)
            attribute2 = self.get_const_cse(attribute2)
            if dma_type_name == "MVIN":
                dma_type = self.get_const_cse(DMA_TYPE[f"{dma_type_name}{self.dma_read_counter}"])
                self.dma_read_counter += 1
                self.dma_read_cache[dma_key] = [dma_type, attribute1, attribute2]
            else:
                dma_type = self.get_const_cse(DMA_TYPE[f"{dma_type_name}{self.dma_write_counter}"])
                # self.dma_write_counter += 1 Is it okay?
                self.dma_write_cache[dma_key] = [dma_type, attribute1, attribute2]
        tag = self.get_tag_cse(tag_name)
        zero_cse = self.get_const_cse(0)

        # Prepare opearnds and attributes
        dram_operand = f"%{dram_var}[%{dram_index_var}]"
        sram_operand = f"%{sram_var}[{sram_index_var}]" # Use string
        tag_var = f"%{tag}[%{zero_cse}]"
        dma_attribute = f"%{attribute1}, %{attribute2}"
        sram_shape = tile_shape
        tag_shape = "memref<1xi32>"

        if dma_type_name == "MVIN":
            src_operand, dst_operand = dram_operand, sram_operand
            src_shape, dst_shape = dram_shape, sram_shape
        else:
            src_operand, dst_operand = sram_operand, dram_operand
            src_shape, dst_shape = sram_shape, dram_shape

        code = f"memref.dma_start {src_operand}, {dst_operand}, %{dma_type}, {tag_var}, {dma_attribute} : {src_shape}, {dst_shape}, {tag_shape}"
        code = code + f" {{padding={padding_type}, sram_stride={tile_stride}}}"
        return code

    def adjust_tile_size(self):
        if self.read_writes is not None:
            read_writes = list(self.read_writes.reads) + list(self.read_writes.writes)
            cv_list = []
            for node in read_writes:
                if len(node) > 1:
                    cv_list.append(self.get_constant_vector2(node[1]))
            max_element = max(cv_list, key=len)
            max_nr_dim = len(max_element)

            sorted_max_element = sorted(max_element, key=lambda x:x[0])
            # Force vector tile size when 3D node is originated from view
            if max_nr_dim == 3 and max_nr_dim != len(self.itervars):
                self.tile_desc.n_col = min(self.tile_desc.get_tile_size(), sorted_max_element[1][0])
                self.tile_desc.n_row = 1
                return

        # Case 1. vector kernel
        if len(self.itervars) == 1:
            tile_size = self.tile_desc.get_tile_size() if self.tile_desc.get_tile_size() < self.ranges[0] else self.ranges[0]
            min_tile_size_unit = self.vector_lane * self.vlen # TODO: VCIX widening is not implemented
            self.tile_desc.n_col = math.ceil(tile_size / min_tile_size_unit) * min_tile_size_unit # padding
            self.tile_desc.n_row = 1
        elif len(self.itervars) == 0:
            self.tile_desc.n_col = 1
            self.tile_desc.n_row = 1

        # Case 2. 2-D tensor (e.g., softmax)
        if len(self.itervars) == 2 and self.reduction_depth == len(self.itervars):
            # Avoid too much padding
            if (self.ranges[0] <= self.vector_lane and self.ranges[0] <= self.tile_desc.n_row):
                self.tile_desc.n_row = self.ranges[0]
                self.tile_desc.used_vector_lane = self.ranges[0]

        # Case 2. 2-D reduction (e.g., batchnorm)
        if len(self.itervars) == 2 and self.reduction_depth == len(self.itervars) - 1:
            if (((self.ranges[0] + 1) // 2) <= self.vector_lane and ((self.ranges[0] + 1) // 2) <= self.tile_desc.n_row):
                self.tile_desc.n_row = ((self.ranges[0] + 1) // 2) * 2
                self.tile_desc.used_vector_lane = (self.ranges[0] + 1) // 2

        # Case 2. 3-D tensor kernel without reduction. Access vector granule!
        if len(self.itervars) == 3 and self.reduction_depth == len(self.itervars):
            self.tile_desc.n_col = self.ranges[-1]
            self.tile_desc.n_row = 1

        # Case 3. N-D tensor kernel with reduction. Not implemented. Need this?
        if len(self.itervars) >= 3 and self.reduction_depth < len(self.itervars):
            raise NotImplementedError()

    def get_scratchpad_buffer(self, dtype, name, tile_size_per_lane, dram_tile_shape, indices, raw_index, is_template=False, buffer=None):
        c_type = mlir_common.DTYPE_TO_C[dtype]
        # Make sure each lane's buffer has at least two element
        tile_size = max(tile_size_per_lane, 2) * self.vector_lane

        if buffer is None:
            buffer = self.loads

        if name not in self.global_vars_dict:
            self.global_vars_dict[name] = list()

        if str(raw_index) not in self.global_vars_dict[name]:
            new_name = f"{name}_{len(self.global_vars_dict[name])}"
            # Add definition to header
            self.header.writeline(f"{c_type} {new_name}_spad[{tile_size // self.vector_lane}] __attribute__ ((section(\".spad\")));")
            self.gem5_header.writeline(f"{c_type} {new_name}_spad[{tile_size}] __attribute__((aligned(64)));")
            self.global_vars.writeline(f"memref.global @{new_name}_spad : {dram_tile_shape}")
            self.global_vars_dict[name].append(str(raw_index))
        else:
            new_name = f"{name}_{self.global_vars_dict[name].index(str(raw_index))}"
        sram_var = self.spad_cse.generate(self.spad_buffer, f"memref.get_global @{new_name}_spad : {dram_tile_shape}")

        zero_cse = self.get_const_cse(0)
        sram_dims = len(dram_tile_shape.split("x")) - 1
        sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)

        return sram_var, indices, sram_index_var

    def get_const_cse(self, value, dtype="index") -> common.CSEVariable:
        # Type convert
        if dtype[0] == "f":
            value = float(value)
        else:
            value = int(value)

        if value not in self.consts:
            self.consts[str(value)+dtype] = self.const_cse.generate(self.const_buffer, f"arith.constant {value} : {dtype}")
        return self.consts[str(value)+dtype]

    def get_tag_cse(self, value, shape="memref<1xi32>"):
        if value not in self.tags:
            self.tags[value] = self.alloc_cse.generate(self.alloc_buffer, f"memref.alloc() : {shape}")
        return self.tags[value]

    def convert_indirect_indexing(self, index :sympy.Expr):
        if "tmp" not in str(index):
            return index

        # Process start
        indirect_dims = [str(dim) for dim in index.free_symbols if "tmp" in str(dim)]
        indirect_dims.sort()
        first_dim = indirect_dims[0]
        spad_vars = dict()
        tmp_comp, self.compute = self.compute, self.dma_loads

        # Load indirect operands
        for target_dim in indirect_dims:
            sram_var, _, tile_numel_per_lane, sram_index_var, tile_shape, vshape = self.spad_buffer_dict[target_dim]
            mlir_dtype = vshape.split("x")[1][:-1]
            vshape = f"vector<{tile_numel_per_lane}x{mlir_dtype}>" # FIXME. Maybe require fine grain compute...
            if tile_numel_per_lane > 1:
                operation = "affine.vector_load"
                line = f"{operation} %{sram_var}[{sram_index_var}] : {tile_shape}, {vshape} // For indirect access"
            else:
                operation = "affine.load"
                line = f"{operation} %{sram_var}[{sram_index_var}] : {tile_shape} // For indirect access"
            out = self.cse.generate(self.dma_loads, line)
            self.register_var_info(out, [tile_numel_per_lane, mlir_dtype])
            spad_vars[target_dim] = out

        # Apply stride
        for arg in index.args:
            if "tmp" not in str(arg):
                continue
            if arg.is_Mul and arg.args[0].is_number:
                coeff_dtype = self.var_info[spad_vars[str(arg.args[1])]][1]
                coeff = ops.constant(int(arg.args[0]), coeff_dtype)
                spad_vars[str(arg.args[1])] = ops.mul(spad_vars[str(arg.args[1])], coeff)
            index = index.replace(arg, 0)

        # Sum
        for dim, var in spad_vars.items():
            if dim == first_dim:
                continue
            spad_vars[first_dim] = ops.add(spad_vars[first_dim], var)

        # Store index var
        sram_var, _, tile_numel_per_lane, sram_index_var, tile_shape, vshape = self.spad_buffer_dict[first_dim]
        mlir_dtype = vshape.split("x")[1][:-1]
        vshape = f"vector<{tile_numel_per_lane}x{mlir_dtype}>" # FIXME. Maybe require fine grain compute...
        if tile_numel_per_lane > 1:
            operation = "affine.vector_store"
            line = f"{operation} %{spad_vars[first_dim]}, %{sram_var}[{sram_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.store"
            line = f"{operation} %{spad_vars[first_dim]}, %{sram_var}[{sram_index_var}] : {tile_shape}"
        out = self.cse.generate(self.dma_loads, line, assignment=False)

        # Conversion
        mlir_dtype = self.var_info[spad_vars[first_dim]][1]
        line = f"affine.load %{sram_var}[{sram_index_var}] : {tile_shape}"
        out = self.cse.generate(self.dma_loads, line)
        if mlir_dtype != "index":
            line = f"arith.index_cast %{out} : {mlir_dtype} to {'index'}"
            out = self.cse.generate(self.dma_loads, line)
        self.register_var_info(out, [1, "index", [1]])
        self.compute = tmp_comp
        return index + sympy.Symbol(str(out))
