import dataclasses
import contextlib
import sympy
import itertools
import re
import os
import math
from functools import reduce
from operator import mul
from typing import List
from typing import Dict
from collections import OrderedDict
import torch
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.codecache import write_atomic, write
from torch._inductor.utils import (
    IndentedBuffer,
    is_welford_reduction,
)
from torch.utils._sympy.functions import ModularIndexing
import PyTorchSimFrontend.extension_codecache as extension_codecache

from . import mlir_common

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
        return "0.0"
    if reduction_type in {"min", "argmin"}:
        return "0.0"
    if reduction_type in {"welford_reduce"}:
        return f"0.0"
    raise AssertionError(reduction_type)

def reduction_combine(reduction_type, start_value, vector_value, tile_size=64):
    if reduction_type == "sum":
        return f"arith.addf %{start_value}, %{vector_value} : vector<{tile_size}xf32>"
    if reduction_type == "prod":
        return f"arith.mulf %{start_value}, %{vector_value} : vector<{tile_size}xf32>"
    if reduction_type == "xor_sum":
        raise NotImplementedError() # TODO: implement
    if reduction_type == "any":
        raise NotImplementedError()
    if reduction_type in ("min", "max"):
        raise NotImplementedError()
    if reduction_type == "welford_reduce":
        raise NotImplementedError()
    if reduction_type == "welford_combine":
        raise NotImplementedError()
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
    index_set = set()
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

        dst_bits = int(dst_mlir_dtype[1:])
        src_bits = int(src_mlir_dtype[1:])
        shape = f"vector<{tile_size}x{dst_mlir_dtype}>" if tile_size > 1 else dst_mlir_dtype
        src_shape = f"vector<{tile_size}x{src_mlir_dtype}>" if tile_size > 1 else src_mlir_dtype
        if dst_mlir_dtype[0] == "i" and src_mlir_dtype[0] == "f":
            raise NotImplementedError("floating point to integer conversion")
        if dst_mlir_dtype[0] == "f" and src_mlir_dtype[0] == "i":
            raise NotImplementedError("integer to floating point conversion")
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

        # if value represented by e notation, convert to float (ex 1e-3 -> 1.0e-3)
        if "e" in str(value):
            value = float(value)
        if src_type[0] == "f":
            value = format(value, ".20f")
        if src_type[0] == "i":
            value = int(value)
        return f'arith.constant {value} : {src_type}', [1, src_type]

    @staticmethod
    def alloc(size, src_type, *args, var_info=None, **kwargs):
        return f"memref.alloc() : memref<{size}x{src_type}>", [size, src_type]

    # transcendental functions
    @staticmethod
    def exp(operand, *args, var_info=None, **kwargs):
        op_type = var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.exp %{operand} : {shape}', [tile_size, dtype]

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
        op_type1 = var_info[operand1]
        op_type2 = var_info[operand2]

        # Type check & auto cast
        if op_type1[1][0] != "f":
            operand1, dtype = ops.to_dtype(operand1, "f32", var_info=var_info)
            var_info[operand1] = dtype

        # Type check & auto cast
        if op_type2[1][0] != "f":
            operand2, dtype = ops.to_dtype(operand2, "f32", var_info=var_info)
            var_info[operand2] = dtype

        op_type1 = var_info[operand1]
        tile_size = op_type1[0]
        dtype = op_type1[1]

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f"math.pow{dtype[0]} %{operand1}, %{operand2} : {shape}", []

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
    def logical_and(operand, *args, var_info=None, **kwargs):
        raise NotImplementedError("logical_and")

    @staticmethod
    def logical_not(operand, *args, var_info=None, **kwargs):
        raise NotImplementedError("logical_not")

    @staticmethod
    def logical_or(operand, *args, var_info=None, **kwargs):
        raise NotImplementedError("logical_not")

    @staticmethod
    def logical_xor(operand, *args, var_info=None, **kwargs):
        raise NotImplementedError("logical_not")

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
        val = ops.constant(0.0, "f32", *args, **kwargs)
        result = ops.where(mask, result, val)
        return result, var_info[result]

    @staticmethod
    def _index_expr(tile_size, buffer, renamed_expression, vec_size, *args, var_info=None, **kwargs):
        strides = [1] * len(tile_size)
        for i in range(len(tile_size) - 2, -1, -1):
            strides[i] = strides[i + 1] * tile_size[i + 1]

        linear_expression = []
        for i, stride in enumerate(strides):
            linear_expression.append(f"d{i}*{stride}")

        dim = ["%d"+str(i) for i in range(len(tile_size))]
        sym_dim = ["d"+str(i) for i in range(len(tile_size))]
        start_dim = [str(0) for i in tile_size]
        end_dim = [str(i) for i in tile_size]

        affine_map_str = "(" + ", ".join(sym_dim) + ") -> ("
        affine_map_str += sympy.printing.ccode(renamed_expression) + ")"

        affine_map_str2 = "(" + ", ".join(sym_dim) + ") -> ("
        affine_map_str2 += "+".join(linear_expression) + ")"

        apply_map_var = f"%index_var = affine.apply affine_map<{affine_map_str}>({', '.join(dim)})\n"
        linear_index_var = f"%buffer_index_var = affine.apply affine_map<{affine_map_str2}>({', '.join(dim)})\n"
        broadcast_var = f"%broadcast_var = vector.broadcast %index_var : index to vector<2xindex>\n"
        affine_store_var = f"affine.vector_store %broadcast_var, %{buffer}[%buffer_index_var] : memref<{vec_size}xindex>, vector<2xindex>\n"

        result = f"affine.parallel ({','.join(dim)}) = ({','.join(start_dim)}) to ({','.join(end_dim)}) {{\n" + \
            apply_map_var + linear_index_var + broadcast_var + affine_store_var + f"}}"
        return result, [None, None]

    @staticmethod
    def index_expr(operand, *args, var_info=None, tile_desc=None, **kwargs):
        # Todo. To support index_expr, we have to custom instructions
        tile_size = tile_desc.get_tile_size()
        if tile_desc.get_used_vlane() != 1:
            raise NotImplementedError("Currently index operation is only executable on single vectorlane configuration")

        vec_size = 1
        for ds in tile_size:
            vec_size *= ds

        buffer = ops.alloc(vec_size, "index")
        ret_type = [vec_size, "index"]

        renamed_symbols = {symbol: "d"+str(symbol)[5:] for symbol in operand.free_symbols}
        renamed_expression = operand.subs(renamed_symbols)
        if operand not in ExtensionOverrides.index_set:
            # Register this operand
            ExtensionOverrides.index_set.add(operand)
            ops._index_expr(tile_size, buffer, renamed_expression, vec_size)

        result = f"affine.vector_load %{buffer}[0] : memref<{vec_size}xindex>, vector<{vec_size}xindex> // {renamed_expression}"
        return result, ret_type

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
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.body = IndentedBuffer()
        self.global_vars = IndentedBuffer()
        self.global_vars_dict = dict()
        self.header = IndentedBuffer()
        self.gem5_header = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.iterator_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="iter")
        self.init_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init")
        self.init_vec_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init_vec")
        self.const_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="const")
        self.alloc_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="alloc")
        self.map_cse = common.CSE("#", self.suffix, name_prefix="map")
        self.consts = dict()
        self.tags = dict()
        self.dma_read_cache = {}
        self.dma_write_cache = {}
        self.dma_read_counter = 1
        self.dma_write_counter = 1
        self.affine_yield = {}
        self.welford_reduce_out = None
        self.reduce_iterator = {}
        self.is_template_kernel = False

    # padding type 0: zero-padding 1: negative-padding(-inf) ...
    def get_padding_type(self):
        ops = self.current_node.node.origins
        if self.current_node.is_reduction():
            for op in ops:
                if "exp" in op.name: # exponential reduciton case
                    return 1
        return 0

    def convert_index(self, expr):
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
        index = self.cse.generate(self.loads, f"affine.apply #{map_var}({args})")
        return index

    def parse_indices(self, expr) -> common.CSEVariable:
        # Constant case
        if expr.is_number:
            return self.get_const_cse(int(expr))

        # Identity case
        if len(expr.args) == 0:
            return expr

        indices = []
        for arg in expr.args:
            if arg.is_Mul and arg.args[0].is_number:
                new_arg = sympy.Symbol(str(self.convert_index(arg.args[1])))
                expr = expr.replace(arg.args[1], new_arg)
            else:
                new_arg = sympy.Symbol(str(self.convert_index(arg)))
                expr = expr.replace(arg, new_arg)
            indices.append(str(new_arg))
        indices.sort()

        # Extract index var
        expr_str = str(expr)
        args = ", ".join(map(str, indices))
        map_var = self.map_cse.generate(self.global_vars, f"affine_map<({args}) -> ({expr_str})>")
        args = ", ".join([f"%{i}" for i in indices])
        index = self.cse.generate(self.loads, f"affine.apply #{map_var}({args})")
        return index

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        padding = self.get_padding_type()
        index_var = self.parse_indices(index)
        dram_var = self.kernel_group.args.input(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        local_tile_desc, index_var = self.get_dma_info(name, index, index_var)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()

        # Define scratch pad buffer
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, self.loads, index_var, index)

        # MVIN Encoding
        code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride, padding)
        self.cse.generate(self.loads, code, assignment = False) # FIXME: assignment = False does not support caching

        # Generate vector load instruction
        operation = "affine.vector_load" if tile_numel_per_lane > 1 else "affine.load"
        shape = f", vector<{tile_numel_per_lane}x{mlir_dtype}>" if tile_numel_per_lane > 1 else ""
        line = f"{operation} %{sram_var}[{sram_index_var}] : {tile_shape}{shape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [tile_numel_per_lane, mlir_dtype])
        return out

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        index_var = self.parse_indices(index)
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        # Prepare dma instruction
        local_tile_desc, index_var = self.get_dma_info(name, index, index_var)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()

        # Define scratch pad buffer
        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, self.stores, index_var, index)

        # Generate vector store instruction
        store_size, operand_type = self.var_info[value]
        operation = "affine.vector_store" if tile_numel_per_lane > 1 and store_size > 1 else "affine.store"
        shape = f", vector<{tile_numel_per_lane}x{mlir_dtype}>" if tile_numel_per_lane > 1 and store_size > 1 else ""
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype, var_info=self.var_info)

        line = f"{operation} %{value}, %{sram_var}[{sram_index_var}] : {tile_shape}{shape}"
        self.cse.generate(self.stores, line, assignment = False)

        # Generate DMA instruction
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.cse.generate(self.stores, code, assignment = False)

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
        else:
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
            acc_var = init
            reduced_shape = type_name
            init = self.cse.generate(self.reduction_prefix, f"arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
            if len(self.ranges) == 1: # 1-D vector to scalar
                axis = "0"
                acc_var = init
                vec_len = self.kernel_group.tile_desc.get_vlane_stride()
                shape = f"vector<{self.var_info[value][0]}x{type_name}>"
                var_info = [vec_len, mlir_common.DTYPE_TO_MLIR[dtype]]
                self.register_var_info(acc, var_info)
            elif len(self.ranges) == 2:
                vec_len = self.kernel_group.tile_desc.get_vlane_stride()
                flattened_size = f"vector<{self.var_info[value][0]}x{type_name}>"

                # It is column majored per lane tile
                expaned_size = f"vector<{self.var_info[value][0]//vec_len}x{vec_len}x{type_name}>"
                value = self.cse.generate(self.compute, f"vector.shape_cast %{value} : {flattened_size} to {expaned_size}")
                shape = expaned_size

                # Edge case for scalar
                if vec_len == 1:
                    reduced_shape = f"{type_name}"
                    init_vec = init
                    axis = "0, 1"
                    acc_var = init
                    var_info = [1, mlir_common.DTYPE_TO_MLIR[dtype]]
                else:
                    reduced_shape = f"vector<{vec_len}x{type_name}>"
                    init_vec = self.cse.generate(self.reduction_prefix, f"vector.broadcast %{init} : {type_name} to {reduced_shape}")
                    axis = "0"
                    acc_var = init_vec
                    var_info = [vec_len, mlir_common.DTYPE_TO_MLIR[dtype]]
                self.register_var_info(acc, var_info)
            else:
                raise NotImplementedError()

            self.reduction_vars[acc] = (reduction_type, iterator, acc_var, reduced_shape)
            out = self.cse.generate(self.compute, reduction_combine_vec(reduction_type, value, iterator, axis=axis, shape=shape, reduced_shape=reduced_shape))
            self.affine_yield[out] = reduced_shape

            self.reduction_cse.reduction_cache[reduction_key] = acc
            self.iterator_cse.reduction_cache[reduction_key] = iterator
            self.init_cse.reduction_cache[reduction_key] = init_vec
        return acc

    def store_reduction(self, name, index, value):
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        index = self.rename_indexing(index)
        index_var = self.parse_indices(index)

        # Tile is always reuduced in inner loop
        local_tile_desc, index_var = self.get_dma_info(name, index, index_var, broadcast=False)
        vlane_split_axis = local_tile_desc.vlane_split_axis
        vlane_stride = local_tile_desc.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()

        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()

        sram_var, index_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, tile_numel_per_lane, tile_shape, self.reductions_suffix, index_var, index)
        if self.welford_reduce_out is not None:
            # raise NotImplementedError()
            sum, sqr_sum, _ = self.welford_reduce_out
            shape = f"vector<{tile_numel_per_lane}x{mlir_dtype}>" if self.buffer_types[name][1] > 1 else mlir_dtype
            # mean
            divider = self.cse.generate(self.reductions_suffix, f"arith.constant {float(self.ranges[self.reduction_depth])} : f32")
            if self.buffer_types[name][1] > 1:
                divider_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %{divider} : f32 to vector<{self.var_info[sum][0]}x{mlir_dtype}>")
            else:
                divider_vec = f"f{self.buffer_types[name][1]}"
            mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sum}, %{divider_vec} : {shape}")

            # m2 = (E(X^2) - E(X)^2) * N
            sqr_mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sqr_sum}, %{divider_vec} : {shape}")
            mean_sqr = self.cse.generate(self.reductions_suffix, f"arith.mulf %{mean}, %{mean} : {shape}")
            variance = self.cse.generate(self.reductions_suffix, f"arith.subf %{sqr_mean}, %{mean_sqr} : {shape}")
            m2 = self.cse.generate(self.reductions_suffix, f"arith.mulf %{variance}, %{divider_vec} : {shape}")
            if self.current_node.node.origin_node: # FIXME: This is a temporary solution
                value = mean
            else:
                value = m2

        # Select mlir store operaiton
        if self.buffer_types[name][1] == 1 or tile_numel_per_lane == 1:
            operation = "affine.store"
            # raise NotImplementedError("Scalar store!")
        else:
            operation =  "affine.vector_store"

        # Select src type
        if tile_numel_per_lane == 1:
            shape = ""
        else:
            shape = f"vector<{tile_numel_per_lane}x{mlir_dtype}>"
            shape = f", {shape}" if self.buffer_types[name][1] > 1 else ""

        line = f"{operation} %{value}, %{sram_var}[{sram_index_var}] : {tile_shape}{shape}"
        self.cse.generate(self.reductions_suffix, line, assignment = False)

        # MVOUT Encoding

        # Generate DMA instruction
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 f"{name}_tag", dram_shape, tile_shape, tile_stride)
        self.cse.generate(self.reductions_suffix, code, assignment = False)

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
        if (self.reduction_depth==0):
            loops = LoopNest([LoopLevel("dummy", 1)])
        reductions.mark_reduction(self.reduction_vars)
        if len(self.affine_yield) > 0:
            vars = ', '.join([f"%{name}" for name, _ in self.affine_yield.items()])
            reduced_shapes = ', '.join([f"{shape}" for _, shape in self.affine_yield.items()])
            self.stores.writeline(f"affine.yield {vars} : {reduced_shapes}")

        code.splice(self.const_buffer)
        code.splice(self.alloc_buffer)
        with contextlib.ExitStack() as stack:
            for loop in loops.loops:
                loop_lines = loop.lines()
                if loop_lines is None:
                    return
                code.writelines(loop_lines)
                stack.enter_context(code.indent(outer_loop=True))
            with contextlib.ExitStack() as stack_outer:
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    for reduction in reductions.loops:
                        reduction_lines = reduction.lines()
                        if reduction_lines is None:
                            return
                        code.writelines(reduction_lines)
                        stack.enter_context(code.indent(outer_loop=False))
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"return")
        return code

    def codegen_nodes(self, nodes, kernel_name):
        src_code = super().codegen_nodes(nodes, kernel_name)

        # Create extra headers for simulators
        write_path = extension_codecache.get_write_path(src_code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        spike_write_path = os.path.join(write_path, "global_var.h")
        gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
        if not os.path.exists(spike_write_path):
            write_atomic(spike_write_path, self.header.getvalue())
        if not os.path.exists(gem5_write_path):
            write_atomic(gem5_write_path, self.gem5_header.getvalue())
        return src_code

    def get_dma_info2(self, name, index):
        current_tile = mlir_common.MLIRTile(self.tile_desc.n_row, self.tile_desc.n_col, self.tile_desc.vector_lane, self.tile_desc.used_vector_lane)
        cv = self.get_constant_vector(index)
        cv2 = self.get_constant_vector2(index)
        tile_size_per_lane = self.tile_desc.get_tile_size_per_lane()            # FIXME. move this
        tile_size_per_lane = 2 if tile_size_per_lane==1 else tile_size_per_lane # Avoid scalar operation

        if len(cv) != len(cv2) and len(cv2) == 3:
            print("Mismatch! ", cv)
            # FIXME. this is really shitty code :(
            cv = cv2#[[1 if x[0] == 0 else x[0], x[1]] for x in cv]

        # Case 0. Tile is 0-D scalar
        if len(cv) == 0:
            # Use only one vectorlane to handle scalar data
            current_tile.n_row = 1
            current_tile.n_col = 1
            current_tile.tile_layout = mlir_common.MLIRTile.TILE_ROW_WISE
            current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_ROW_WISE
            mm_stride, tile_size_per_lane = 1, 1
            vlane_stride = current_tile.get_vlane_stride()
        # Case 1. Tile is 1-D vector type
        elif len(cv) == 1 and len(cv) <= self.reduction_depth:
            current_tile.n_row = 1
            current_tile.n_col = self.tile_desc.get_tile_size()
            current_tile.tile_layout = mlir_common.MLIRTile.TILE_ROW_WISE
            current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE # Actually it is not needed in vector case
            vlane_stride = current_tile.get_vlane_stride()
            mm_stride = current_tile.n_col
            if self.is_scalar(name): # scalar to vector broadcasting
                mm_stride = 0
                current_tile.n_row, current_tile.n_col = current_tile.n_col, current_tile.n_row
        # Case 2. Tile is 1-D vector type with reduction
        elif len(cv) == 1 and len(cv) == self.reduction_depth + 1:
            # Use only one vectorlane to reduce a vector
            current_tile.tile_layout = mlir_common.MLIRTile.TILE_ROW_WISE
            current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_ROW_WISE
            current_tile.n_row = 1
            current_tile.n_col = self.tile_desc.get_tile_size()
            current_tile.used_vector_lane = 1
            vlane_stride = current_tile.get_vlane_stride()
            mm_stride = 0 # don't care
            tile_size_per_lane = current_tile.get_tile_size_per_lane()
            if self.is_scalar(name): # scalar to vector broadcasting
                current_tile.n_row, current_tile.n_col = current_tile.n_col, current_tile.n_row
        # Case 3. Tile is 2-D tile
        elif len(cv) == 2:
            is_reduction = self.reduction_depth == 1
            if cv[0][0] != 0 and cv[1][0] != 0:
                is_transposed = cv[0][0] < cv[1][0]
                if is_transposed:
                    current_tile.n_row = self.tile_desc.n_col
                    current_tile.n_col = self.tile_desc.n_row
                    mm_stride = self.ranges[0]
                else:
                    current_tile.n_row = self.tile_desc.n_row
                    current_tile.n_col = self.tile_desc.n_col
                    mm_stride = self.ranges[1]

                if is_reduction and is_transposed:
                    current_tile.tile_layout = mlir_common.MLIRTile.TILE_COL_WISE
                    current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_ROW_WISE
                    vlane_stride = current_tile.get_vlane_stride()
                elif is_reduction and not is_transposed:
                    current_tile.tile_layout = mlir_common.MLIRTile.TILE_ROW_WISE
                    current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE
                    vlane_stride = current_tile.get_vlane_stride()
                elif not is_reduction and is_transposed:
                    # Transposed case
                    current_tile.tile_layout = mlir_common.MLIRTile.TILE_COL_WISE
                    current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE
                    vlane_stride = current_tile.get_vlane_stride()
                else: # not is_reduction and not is_transpose
                    current_tile.tile_layout = mlir_common.MLIRTile.TILE_COL_WISE if self.tile_desc.vector_lane_axis else mlir_common.MLIRTile.TILE_ROW_WISE
                    current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_ROW_WISE
                    vlane_stride = current_tile.get_vlane_stride()
            else:
                # Broadcast pattern
                current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_ROW_WISE
                mm_stride = 0
                if cv[0][0] == 0:
                    current_tile.tile_layout = mlir_common.MLIRTile.TILE_COL_WISE if self.tile_desc.vector_lane_axis else mlir_common.MLIRTile.TILE_ROW_WISE
                    current_tile.n_row = self.tile_desc.n_row
                    current_tile.n_col = self.tile_desc.n_col
                    vlane_stride = current_tile.get_vlane_stride()
                else: # cv[1][0] == 0
                    current_tile.n_row = self.tile_desc.n_col
                    current_tile.n_col = self.tile_desc.n_row
                    vlane_stride = current_tile.get_cols_per_lane()
                    if not is_reduction:
                        current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE
                        vlane_stride = current_tile.n_col if self.tile_desc.vector_lane_axis else vlane_stride
        elif len(cv) == 3:
            current_tile.tile_per_lane_layout = mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE # Actually it is not needed in vector case
            mm_stride = cv[-1][0]
            # When current_tile.n_col stride is 1, we can access row vector
            if mm_stride == 1:
                current_tile.n_row = 1
                current_tile.n_col = self.tile_desc.get_tile_size()
            # if current_tile.n_col stride is not 1, we have to access in a column vector
            else:
                current_tile.n_row = self.tile_desc.get_tile_size()
                current_tile.n_col = 1
            vlane_stride = current_tile.get_tile_size_per_lane()
        else:
            raise NotImplementedError()

        #assert(not (dtype==torch.bool and vlane_stride < 8))
        vlane_split_axis = int(current_tile.tile_per_lane_layout == mlir_common.MLIRTile.TILE_PER_LANE_COL_WISE)
        return vlane_split_axis, vlane_stride, [current_tile.n_row, current_tile.n_col], tile_size_per_lane

    def get_dma_info(self, name, index, index_var, broadcast=True): # Need more argument?
        """
        A tile descriptor exists that is configured on a kernel group
        DMA desc should be adjusted according to buffer.
        Therefore, this function shoulde determin DRAM, SRAM stride and
        vectorlane mapping policy
        """
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

        # Reduction can have two type of tile size
        if broadcast and (total_dims != local_dims or (self.reduction_depth!=len(total_dims) and total_dims[:self.reduction_depth] == local_dims)):
            # We have to create custom apply map to provide dram stride
            # ex) (d0, d1, ... dn, dn+1, dn+2, dk) -> (s0*d0 + s1*d1 + ... dn*0+ dn+1*0 + ... dk*0 + const)
            fake_dim = self.get_const_cse(0)
            input_expr = ",".join(["d"+str(i) for i in total_dims])
            output_expr = str(index).replace('index', 'd')
            input_argument = ",".join(["%index" + str(i) if i in local_dims else f"%{fake_dim}" for i in total_dims])
            map_var = self.map_cse.generate(self.global_vars, f"affine_map<({input_expr}) -> ({output_expr})>")
            index_var = self.cse.generate(self.loads, f"affine.apply #{map_var}({input_argument})")
            local_dims = total_dims # Brodatcast tile shape

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
            local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(local_dims[0])])
            local_tile_desc.vlane_split_axis = 0
            local_tile_desc.vlane_stride = kg_tile_desc.get_dim_size(local_dims[0])
        # Case 3. Tile is 2-D tile
        elif len(local_dims) == 2:
            is_reduction = self.reduction_depth == 1
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
            is_reduction = self.reduction_depth < 3
            if is_reduction:
                #local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in dims], [1, 0])
                #local_tile_desc.vlane_split_axis = local_vlane_split_axis
                #local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
                raise NotImplementedError("Currently not implemented... ;)")
            else:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
                local_tile_desc.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vlane_stride = kg_tile_desc.vlane_stride
        else:
            raise NotImplementedError("Currently not implemented... ;)")

        if len(implicit_local_dims)!=0 and len(local_dims) != len(implicit_local_dims):
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

    def get_scratchpad_buffer(self, dtype, name, tile_size_per_lane, dram_tile_shape, code_buffer, indices, raw_index, is_template=False):
        c_type = mlir_common.DTYPE_TO_C[dtype]
        # Make sure each lane's buffer has at least two element
        tile_size = max(tile_size_per_lane, 2) * self.vector_lane

        if dtype == torch.bool and not is_template:
            mapping = self.map_cse.generate(self.global_vars, f"affine_map<({indices}) -> ({indices} floordiv 8)>")
            indices = self.cse.generate(self.loads, f"affine.apply #{mapping}(%{indices})") # FIXME. Only loads?

        if name not in self.global_vars_dict:
            self.global_vars_dict[name] = list()

        if str(raw_index) not in self.global_vars_dict[name]:
            new_name = f"{name}_{len(self.global_vars_dict[name])}"
            # Add definition to header
            self.header.writeline(f"{c_type} {new_name}_spad[{tile_size // self.vector_lane}] __attribute__ ((section(\".spad\")));")
            self.gem5_header.writeline(f"{c_type} {new_name}_spad[{tile_size}];")
            self.global_vars.writeline(f"memref.global @{new_name}_spad : {dram_tile_shape}")
            self.global_vars_dict[name].append(str(raw_index))
        else:
            new_name = f"{name}_{self.global_vars_dict[name].index(str(raw_index))}"
        buffer = self.cse.generate(code_buffer, f"memref.get_global @{new_name}_spad : {dram_tile_shape}")

        zero_cse = self.get_const_cse(0)
        sram_dims = len(dram_tile_shape.split("x")) - 1
        sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)

        return buffer, indices, sram_index_var

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

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    start: int = 0
    step: int = 1
    reduction_vars: Dict[str, str] = None

    def lines(self):
        if self.reduction_vars:
            acc = ', '.join([f"%{acc.name}" for acc in self.reduction_vars.keys()])
            args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
            dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
            line = f"{acc} = affine.for %{self.var} = {self.start} to {self.size} step {self.step} iter_args({args}) -> ({dtype})"
        else:
            line = f"affine.for %{self.var} = {self.start} to {self.size} step {self.step}"

        return [line]

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd