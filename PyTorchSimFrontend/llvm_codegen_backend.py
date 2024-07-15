import dataclasses
import contextlib
import sympy
import itertools
import re
from functools import reduce
from operator import mul
from typing import List
from typing import Dict
import torch
from torch._inductor import dependencies
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.utils import IndentedBuffer
import extension_codecache

from . import llvm_common
from . import llvm_lowering

def reduction_alloc(code, stack, vars):
    # FIXME. USE VARIABLES' TYPE...
    REDUCTION_TYPE = "float"
    REDUCTION_SIZE = 4
    for var in vars:
        line = f"%{var} = alloca {REDUCTION_TYPE}, align {REDUCTION_SIZE}"
        code.writeline(line)

def matrix_reduction_alloc(code, stack, vars, tile_row):
    # FIXME. USE VARIABLES' TYPE...
    REDUCTION_TYPE = "float"
    REDUCTION_SIZE = 4
    for var in vars:
        line = f"%{var} = alloca {REDUCTION_TYPE}, i32 {tile_row}, align {REDUCTION_SIZE}"
        code.writeline(line)

def reduction_init(reduction_type, dtype):
    if dtype in cpp.DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return "0.0"
    if reduction_type == "prod":
        return "1.0"
    if reduction_type in {"max", "argmax"}:
        return "0.0"
    if reduction_type in {"min", "argmin"}:
        return "0.0"
    raise AssertionError(reduction_type)

def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"fadd float %{var}, %{next_value}"
    if reduction_type == "prod":
        return f"fmul float %{var}, %{next_value}"
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

def vector_reduction_combine(reduction_type, start_value, vector_value):
    if reduction_type == "sum":
        return f"tail call float @llvm.vector.reduce.fadd.nxv2f32(float %{start_value}, <vscale x 2 x float> %{vector_value})"
    if reduction_type == "prod":
        return f"tail call float @llvm.vector.reduce.fmul.nxv2f32(float %{start_value}, <vscale x 2 x float> %{vector_value})"
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

def matrix_reduction_combine(reduction_type, start_value, vector_value, tile_row=64):
    if reduction_type == "sum":
        return f"fadd <{tile_row} x float> %{start_value}, %{vector_value}"
    if reduction_type == "prod":
        return f"fmul <{tile_row} x float> %{start_value}, %{vector_value}"
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

def matrix_partial_reduction_combine(reduction_type, vector_value, tile_row=64):
    if reduction_type == "sum":
        return f"tail call float @llvm.vector.reduce.fadd.nxv2f32(float 0.0, <{tile_row} x float> %{vector_value})"
    if reduction_type == "prod":
        return f"tail call float @llvm.vector.reduce.fmul.nxv2f32(float 1.0, <{tile_row} x float> %{vector_value})"
    if reduction_type in ("min", "max"):
        return f"tail call float @llvm.vector.reduce.f{reduction_type}.nxv2f32(<{tile_row} x float> %{vector_value})"
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

            """
        )

class ExtensionOverrides(common.OpOverrides):
    """Map element-wise ops to LLVM IR"""

    @staticmethod
    def add(operand1, operand2, **kwargs):
        return f'fadd float %{operand1}, %{operand2}' # TODO: separate float and integer

    @staticmethod
    def sub(operand1, operand2, **kwargs):
        return f'fsub float %{operand1}, %{operand2}'

    @staticmethod
    def mul(operand1, operand2, **kwargs):
        return f'fmul float %{operand1}, %{operand2}'

    @staticmethod
    def div(operand1, operand2, **kwargs):
        return f'fdiv float %{operand1}, %{operand2}'

class VectorOverrides(ExtensionOverrides):
    @staticmethod
    def vector_add(operand1, operand2, **kwargs):
        return f'fadd <vscale x 2 x float> %{operand1}, %{operand2}'

    @staticmethod
    def vector_sub(operand1, operand2, **kwargs):
        return f'fsub <vscale x 2 x float> %{operand1}, %{operand2}'

    @staticmethod
    def vector_mul(operand1, operand2, **kwargs):
        return f'fmul <vscale x 2 x float> %{operand1}, %{operand2}'

    @staticmethod
    def vector_div(operand1, operand2, **kwargs):
        return f'fdiv <vscale x 2 x float> %{operand1}, %{operand2}'

class MatrixOverrides(ExtensionOverrides):
    @staticmethod
    def add(operand1, operand2, tile_size=16):
        return f'fadd <{tile_size} x float> %{operand1}, %{operand2}'

    @staticmethod
    def sub(operand1, operand2, tile_size=4):
        return f'fsub <{tile_size} x float> %{operand1}, %{operand2}'

    @staticmethod
    def mul(operand1, operand2, tile_size=4):
        return f'fmul <{tile_size} x float> %{operand1}, %{operand2}'

    @staticmethod
    def div(operand1, operand2, tile_size=4):
        return f'fdiv <{tile_size} x float> %{operand1}, %{operand2}'

    @staticmethod
    def truediv(operand1, operand2, tile_size=4):
        return f'fdiv <{tile_size} x float> %{operand1}, %{operand2}'

    @staticmethod
    def constant(value, dtype, tile_size=4):
        return f'insertelement <1 x float> undef, float {value}, i32 0'

    @staticmethod
    def exp(operand, tile_size=4):
        return f'tail call <{tile_size} x float> @llvm.exp.f32(<{tile_size} x float> %{operand})'

    @staticmethod
    def maximum(operand1, operand2, tile_size=4):
        return f'tail call <{tile_size} x float> @llvm.maximum.f32(<{tile_size} x float> %{operand1}, <{tile_size} x float >%{operand2})'

    @staticmethod
    def relu(x, tile_size=4):
        return ops.maximum(x, ops.constant(0.0, torch.int32))

SYMPY_TO_LLVM = {
    sympy.core.mul.Mul: "mul",
    sympy.core.add.Add: "add",
}

class LLVMKernel(llvm_common.BaseLLVMKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, args=None):
        super().__init__(llvm_common.LLVMKernelArgs())
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.index_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_idx")
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.tiling_indices = [0, 1]

    def get_constant_vector(self, expr):
        constant_vector = [int(expr.coeff(var)) for var in self.itervars]
        return constant_vector

    def add_desc(self, is_load, base_addr, element_size, stride_list, tile_size):
        if is_load:
            key = f"load{len(self.load_desc)}"
            self.load_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }
        else:
            key = f"store{len(self.store_desc)}"
            self.store_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }

    def depth_first_traverse(self, expr, buffer, cse):
        child_var = []
        for arg in expr.args:
            child_var.append(self.depth_first_traverse(arg, buffer, cse))

        while len(child_var) >= 3:
            first = child_var.pop(0)
            second = child_var.pop(0)
            first_prefix = "" if first.is_number else "%"
            second_prefix = "" if second.is_number else "%"

            line = f"{SYMPY_TO_LLVM[expr.func]} nsw i64 {first_prefix}{first}, {second_prefix}{second}"
            var = cse.generate(buffer, line)
            var = sympy.symbols(f"{var}")
            child_var.append(var)

        if len(expr.args) == 0:
            return expr

        elif len(child_var) == 2:
            first = child_var[1]
            second = child_var[0]
            first_prefix = "" if first.is_number else "%"
            second_prefix = "" if second.is_number else "%"
            line = f"{SYMPY_TO_LLVM[expr.func]} nsw i64 {first_prefix}{first}, {second_prefix}{second}"
            var = cse.generate(buffer, line)
            var = sympy.symbols(f"{var}")
            return var
        else:
            raise Exception()

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        def select_tiling_indices():
            all_index = []
            for node in nodes:
                rw = dependencies.extract_read_writes(node._body, *node._sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
            for index in all_index:
                for var in index.free_symbols:
                    if not re.search(r"^d\d+$", var.name):
                        continue
                    stride = cpp.stride_at(var, index)
                    if stride == 1:
                        contig_vars.add(int(var.name[1:]))
                        contig_vars_list.append(int(var.name[1:]))
                    elif all(s.name.startswith("s") for s in stride.free_symbols):
                        non_contig_stride_const.add(int(var.name[1:]))
                    else:
                        non_contig_stride_other.add(int(var.name[1:]))
            contig_only = (
                contig_vars - non_contig_stride_const - non_contig_stride_other
            )
            if len(contig_vars) == 0:
                # no contiguous vars
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (
                contig_vars & non_contig_stride_const
            ) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if (
                len(contig_vars_sorted) == 2
                and contig_vars_sorted[-1] in contig_and_const_stride
                and contig_vars_sorted[-1] == len(self.itervars) - 1
            ):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]

        self.set_ranges(group, reduction_group)
        self.tiling_indices = select_tiling_indices()

        with self as kernel:
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                node.run(vars, reduction_vars)

        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()
        return src_code

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        index = self.depth_first_traverse(index, self.loads, self.index_cse)
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.cse.generate(self.loads, line)
        line = f"load {type_name}, ptr %{var}, align {align}"
        return self.cse.generate(self.loads, line)

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        index = self.depth_first_traverse(index, self.stores, self.index_cse)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.cse.generate(self.stores, line)
        if (isinstance(value, list)):
            value = value[1]
        line = f"store {type_name} %{value}, ptr %{var}, align {align}"
        self.cse.generate(self.stores, line, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        else:
            reduction_key = src_dtype, reduction_type, value
            acc = self.reduction_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            self.reduction_vars[acc] = reduction_type
            type_name = llvm_common.DTYPE_TO_LLVM[dtype]
            align = llvm_common.DTYPE_SIZE[dtype]
            self.reduction_prefix.writeline(f"store {type_name} {reduction_init(reduction_type, dtype)}, ptr %{acc}, align {align}")
            line = f"load {type_name}, ptr %{acc}, align {align}"

            # NOTE. To keep below line be under the compute, used store buffers
            temp = self.cse.generate(self.stores, line)
            output = self.cse.generate(self.stores, reduction_combine(reduction_type, temp, value))
            line = f"store {type_name} %{output}, ptr %{acc}, align {align}"
            self.cse.generate(self.stores, line, assignment = False)
            self.reduction_cse.reduction_cache[reduction_key] = acc
        return acc

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        index = self.depth_first_traverse(index, self.reduction_suffix, self.index_cse)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"load {type_name}, ptr %{value}, align {align}"
        value = self.reduction_cse.generate(self.reductions_suffix, line)
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.cse.generate(self.reductions_suffix, line)
        line = f"store {type_name} %{value}, ptr %{var}, align {align}"
        self.cse.generate(self.reductions_suffix, line, assignment = False)

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop body part
        loops = [LoopLevel(var, size, idx) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            if self.reduction_vars:
                reduction_alloc(code, stack, self.reduction_vars)
            loops.codegen(code, stack)
            with contextlib.ExitStack() as stack_outer:
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    reductions.codegen(code, stack)
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"ret void")
        return code

    def codegen_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        arg_defs, _, _ = self.args.llvm_argdefs()
        code = self._codegen_kernel(arg_defs, kernel_name)
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes = self.args.llvm_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")


    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_attributes = self.args.llvm_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def _codegen_kernel(self, arg_defs, kernel_name):
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        # Todo. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'define void @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())
        code.writeline(f'declare i64 @llvm.umax.i64(i64, i64) #1')
        code.writeline(f'declare i32 @llvm.umax.i32(i32, i32) #1')
        return code


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

class VectorizedLLVMKernel(LLVMKernel):
    overrides = VectorOverrides

    def __init__(self):
        super().__init__()
        self.vector_loads = IndentedBuffer()
        self.vector_stores = IndentedBuffer()
        self.vector_index_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_vec_idx")
        self.vector_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="vector_body")
        self.vector_reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")

    def load(self, name: str, index: sympy.Expr):
        scalar_var = super().load(name, index)
        index = self.rename_indexing(index)
        index = index.replace(sympy.symbols(f"index{len(self.itervars)-1}"), sympy.symbols(f"vector.index{len(self.itervars)-1}"))
        index = self.depth_first_traverse(index, self.vector_loads, self.vector_index_cse)
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.vector_cse.generate(self.vector_loads, line)

        # NOTE. Since clang 16.0 always used this constant, 2 is hard coded
        line = f"load <vscale x 2 x {type_name}>, ptr %{var}, align {align}"
        return [self.vector_cse.generate(self.vector_loads, line), scalar_var]

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        super().store(name, index, value, *args, **kwargs)
        index = self.rename_indexing(index)
        index = index.replace(sympy.symbols(f"index{len(self.itervars)-1}"), sympy.symbols(f"vector.index{len(self.itervars)-1}"))
        index = self.depth_first_traverse(index, self.vector_stores, self.vector_index_cse)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.vector_cse.generate(self.vector_stores, line)

        # NOTE. Since clang 16.0 always used this constant, 2 is hard coded
        if (isinstance(value, list)):
            value = value[0]
        line = f"store <vscale x 2 x {type_name}> %{value}, ptr %{var}, align {align}"
        self.vector_cse.generate(self.vector_stores, line, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        super().reduction(dtype, src_dtype, reduction_type, value[1])
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        else:
            reduction_key = src_dtype, reduction_type, value
            acc = self.vector_reduction_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            type_name = llvm_common.DTYPE_TO_LLVM[dtype]
            align = llvm_common.DTYPE_SIZE[dtype]
            line = f"load {type_name}, ptr %{acc}, align {align}"

            # NOTE. To keep lines below under the compute lines, used store buffers
            temp = self.vector_cse.generate(self.vector_stores, line)
            output = self.vector_cse.generate(self.vector_stores, vector_reduction_combine(reduction_type, temp, value[0]))
            line = f"store {type_name} %{output}, ptr %{acc}, align {align}"
            self.vector_cse.generate(self.vector_stores, line, assignment = False)
        return acc

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop arguments
        loops_args = [[var, size, idx] for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]

        # Loop initialize
        loops_list = [LoopLevel(*args) for args in loops_args[:-1]]
        vector_loops_list = [VectorLoopLevel(*loops_args[-1])]
        scalar_loop = [LoopLevel(loops_args[-1][0], loops_args[-1][1], loops_args[-1][2], f"%scalar.index.ph")]

        loops = LoopNest(loops_list[: self.reduction_depth])
        reductions = LoopNest(loops_list[self.reduction_depth :])
        inner_most = LoopNest(vector_loops_list)
        inner_most_scalar = LoopNest(scalar_loop)

        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            if self.reduction_vars:
                reduction_alloc(code, stack, self.reduction_vars)
            loops.codegen(code, stack)
            code.splice(self.reduction_prefix)
            with contextlib.ExitStack() as stack:
                reductions.codegen(code, stack)
                with contextlib.ExitStack() as stack_inner:
                    inner_most.codegen(code,stack_inner)
                    code.splice(self.vector_loads)
                    code.splice(self.vector_compute)
                    code.splice(self.vector_stores)

                with contextlib.ExitStack() as stack_inner:
                    inner_most_scalar.codegen(code, stack_inner)
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"ret void")
        return code

    def _codegen_kernel(self, arg_defs, kernel_name):
        code = super()._codegen_kernel(arg_defs, kernel_name)
        # Add vector llvm intrinsics definition
        code.writeline(f'declare i64 @llvm.vscale.i64() #2')
        code.writeline(f'declare i32 @llvm.vscale.i32() #2')
        return code

class MatrixLLVMKernel(LLVMKernel):
    overrides = MatrixOverrides

    def __init__(self):
        super().__init__()
        # Defaulat tile setting
        self.tile_row = 64
        self.tile_col = 64
        self.tile_size = self.tile_row * self.tile_col

    def get_load_info(self, vec_len, cv):
        tile_row = self.tile_row
        tile_col = self.tile_col
        if vec_len > 1:
            stride = reduce(mul, cv, 1)
            if len(cv) == 2 and cv[1] == 0: # if the tile is row major vector
                vec_len, tile_col, stride = self.tile_row, 1, self.tile_row
            elif len(cv) == 2 and cv[0] == 0: # if the tile is colum major vector
                vec_len, tile_row, stride = self.tile_col, 1, 1
            elif len(cv) == 1: # if the tile is vector
                stride = tile_row
        elif vec_len == 1: # scalar
            tile_row, tile_col, stride = 1, 1, 1
        return tile_row, tile_col, stride, vec_len

    def need_vec_transpose(self, index):
        return (
            len(self.itervars) > 1
            and self.outer_idx is not None
            and cpp.stride_at(self.itervars[self.outer_idx], index) == 1
            and index.has(self.itervars[self.tiling_idx])
            and not cpp.stride_at(self.itervars[self.tiling_idx], index).has(
                self.itervars[self.tiling_idx]
            )
            and not cpp.stride_at(self.itervars[self.tiling_idx], index).has(
                self.itervars[self.outer_idx]
            )
        )

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]

        index = self.rename_indexing(index)
        cv = self.get_constant_vector(index)
        self.add_desc(True, name, align, cv, [self.tile_col, self.tile_row])
        new_index = self.depth_first_traverse(index, self.loads, self.index_cse)
        vec_len = self.tile_size if not new_index.is_number else 1
        new_index = f"%{new_index}" if not new_index.is_number else new_index
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 {new_index}"
        var = self.cse.generate(self.loads, line)
        tile_row, tile_col, stride, vec_len = self.get_load_info(vec_len, cv)
        line = f"call <{vec_len} x {type_name}> @llvm.matrix.column.major.load.v{vec_len}f32.p0f32(ptr %{var}, i64 {stride}, i1 0, i32 {tile_row}, i32 {tile_col})"
        out_var = self.cse.generate(self.loads, line)
        if self.need_vec_transpose(index):
            line = f"call <{vec_len} x {type_name}> @llvm.matrix.transpose.v{vec_len}f32.p0f32(<{vec_len} x {type_name}> %{out_var}, i32 {tile_row}, i32 {tile_col})"
            out_var = self.cse.generate(self.loads, line)
        self.tile_shape[out_var] = [tile_row, tile_col]
        return out_var

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]

        index = self.rename_indexing(index)
        cv = self.get_constant_vector(index)
        self.add_desc(False, name, align, cv, [self.tile_col, self.tile_row])
        new_index = self.depth_first_traverse(index, self.stores, self.index_cse)
        vec_len = self.tile_size if not new_index.is_number else 1
        new_index = f"%{new_index}" if not new_index.is_number else new_index
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 {new_index}"
        var = self.cse.generate(self.stores, line)
        if (isinstance(value, list)):
            value = value[0]
        if self.need_vec_transpose(index):
            line = f"call <{vec_len} x {type_name}> @llvm.matrix.transpose.v{vec_len}f32.p0f32(<{vec_len} x {type_name}> %{value}, i32 {self.tile_row}, i32 {self.tile_col})"
            value = self.cse.generate(self.stores, line)
        if vec_len > 1:
            stride = self.ranges[-1]
            line = f"call void @llvm.matrix.column.major.store.v{self.tile_size}f32.p0f32(<{self.tile_size} x {type_name}> %{value}, ptr %{var}, i64 {stride}, i1 0, i32 {self.tile_row}, i32 {self.tile_col})"
        elif vec_len == 1:
            line = f"call void @llvm.matrix.column.major.store.v1f32.p0f32(<1 x {type_name}> %{value}, ptr %{var}, i64 1, i1 0, i32 1, i32 1)"
        self.cse.generate(self.stores, line, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        else:
            reduction_key = src_dtype, reduction_type, value
            acc = self.reduction_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            self.reduction_vars[acc] = reduction_type
            type_name = llvm_common.DTYPE_TO_LLVM[dtype]
            align = llvm_common.DTYPE_SIZE[dtype]
            comma = ", "
            if (self.tile_col == 1): # if the tile is vector
                self.reduction_prefix.writeline(f"store {type_name} {reduction_init(reduction_type, dtype)}, ptr %{acc}, align {align}")
                line = f"load <{self.tile_col} x {type_name}>, ptr %{acc}, align {align}"
            else:
                array_line = [f"{type_name} {reduction_init(reduction_type, dtype)}" for _ in range(self.tile_row)]
                self.reduction_prefix.writeline(f"store <{self.tile_row} x {type_name}> <{comma.join(array_line)}>, ptr %{acc}, align {align}")
                line = f"load <{self.tile_row} x {type_name}>, ptr %{acc}, align {align}"

            # NOTE. To keep below line be under the compute, used store buffers
            temp = self.cse.generate(self.stores, line)
            if self.tiling_idx >= self.reduction_depth: # horizontal reduction
                output = []
                for i in range(self.tile_col):
                    if self.tile_col != 1:
                        indexes = [f"i32 {i*self.tile_row+j}" for j in range(self.tile_row)]
                        line = f"shufflevector <{self.tile_size} x {type_name}> %{value}, <{self.tile_size} x {type_name}> undef, <{self.tile_row} x i32> <{comma.join(indexes)}>"
                        split_vector = self.cse.generate(self.stores, line)
                        reduced_vector = self.cse.generate(self.stores, matrix_partial_reduction_combine(reduction_type, split_vector, self.tile_row))
                    else:
                        reduced_vector = self.cse.generate(self.stores, matrix_partial_reduction_combine(reduction_type, value, self.tile_row))
                    output.append(reduced_vector)
                length = len(output)
                size = 1
                while(len(output) > 1):
                    op1 = output.pop(0)
                    op2 = output.pop(0)
                    if size == 1:
                        line = f"insertelement <2 x {type_name}> undef, {type_name} %{op1}, i32 0"
                        temp_vec = self.cse.generate(self.stores, line)
                        line = f"insertelement <2 x {type_name}> %{temp_vec}, {type_name} %{op2}, i32 1"
                    else:
                        indexes = [f"i32 {j}" for j in range(size * 2)]
                        line = f"shufflevector <{size} x {type_name}> %{op1}, <{size} x {type_name}> %{op2}, <{size * 2} x i32> <{comma.join(indexes)}>"
                    out = self.cse.generate(self.stores, line)
                    output.append(out)
                    if (len(output) == length / 2):
                        size *= 2
                        length = len(output)
                if (self.tile_col == 1):
                    line = f"insertelement <{self.tile_col} x {type_name}> undef, {type_name} %{output[0]}, i32 0"
                    cast_vec = self.cse.generate(self.stores, line)
                    line = f"fadd <{self.tile_col} x {type_name}> %{temp}, %{cast_vec}"
                else:
                    line = f"fadd <{self.tile_row} x {type_name}> %{temp}, %{output[0]}"
                stored_vector = self.cse.generate(self.stores, line)
            else:
                partial_value = temp
                for i in range(self.tile_col):
                    if self.tile_col != 1:
                        indexes = [f"i32 {i*self.tile_row+j}" for j in range(self.tile_row)]
                        line = f"shufflevector <{self.tile_size} x {type_name}> %{value}, <{self.tile_size} x {type_name}> undef, <{self.tile_row} x i32> <{comma.join(indexes)}>"
                        split_vector = self.cse.generate(self.stores, line)
                        partial_value = self.cse.generate(self.stores, matrix_reduction_combine(reduction_type, partial_value, split_vector, self.tile_row))
                    else:
                        partial_value = self.cse.generate(self.stores, matrix_reduction_combine(reduction_type, partial_value, value, self.tile_row))
                stored_vector = partial_value
            if (self.tile_col == 1):
                line = f"store <{self.tile_col} x {type_name}> %{stored_vector}, ptr %{acc}, align {align}"
            else:
                line = f"store <{self.tile_row} x {type_name}> %{stored_vector}, ptr %{acc}, align {align}"
            self.cse.generate(self.stores, line, assignment = False)
            self.reduction_cse.reduction_cache[reduction_key] = acc
        return acc

    def store_reduction(self, name, index, value):
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]

        index = self.rename_indexing(index)
        cv = self.get_constant_vector(index)
        self.add_desc(False, name, align, cv, [1, self.tile_row])
        index = self.depth_first_traverse(index, self.reduction_suffix, self.index_cse)
        if (self.tile_col == 1):
            line = f"load <{self.tile_col} x {type_name}>, ptr %{value}, align {align}"
        else:
            line = f"load <{self.tile_row} x {type_name}>, ptr %{value}, align {align}"
        value = self.reduction_cse.generate(self.reductions_suffix, line)
        if (self.tile_col == 1):
            line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 0"
        else:
            line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.cse.generate(self.reductions_suffix, line)
        stride = self.ranges[-1]
        if (self.tile_col == 1):
            line = f"call void @llvm.matrix.column.major.store.v{self.tile_col}f32.p0f32(<{self.tile_col} x {type_name}> %{value}, ptr %{var}, i64 1, i1 0, i32 {self.tile_col}, i32 1)"
        else:
            line = f"call void @llvm.matrix.column.major.store.v{self.tile_row}f32.p0f32(<{self.tile_row} x {type_name}> %{value}, ptr %{var}, i64 {stride}, i1 0, i32 {self.tile_row}, i32 1)"
        self.cse.generate(self.reductions_suffix, line, assignment = False)

    def codegen_loops(self):
        code = common.BracesBuffer()
        self.loop_info = {}
        # Loop body part
        loops_args = [[var, size, idx] for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
        outer_loops = [LoopLevel(var, size, idx) for var, size, idx in loops_args[:-2]]
        loops = [MatrixLoopLevel(var, size, idx, tile_row=self.tile_row) for var, size, idx in loops_args[-2:]]
        loops = outer_loops + loops
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            if self.reduction_vars:
                matrix_reduction_alloc(code, stack, self.reduction_vars, self.tile_row)
            self.loop_info.update(loops.codegen(code, stack))
            with contextlib.ExitStack() as stack_outer:
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    self.loop_info.update(reductions.codegen(code, stack))
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"ret void")
        return code

    def set_ranges(self, lengths, reduction_lengths):
        ret = super().set_ranges(lengths, reduction_lengths)
        # do vertical reduction as the tail loop
        if len(self.itervars) > 1:
            if len(self.tiling_indices) == 1:
                self.tiling_idx = self.tiling_indices[0]
                self.outer_idx = None
            else:
                self.outer_idx, self.tiling_idx = (
                    self.tiling_indices
                    if self.tiling_indices[1] < self.reduction_depth
                    else reversed(self.tiling_indices)
                )
        else:
            self.tiling_idx = self.tiling_indices[0]
            self.outer_idx = None

        # FIXME. this doesn't look pretty...
        # We have to change this logic to configurable tile_size
        if len(self.itervars) == 1:
            self.tile_row = min(self.tile_size, self.ranges[0])
            self.tile_col = 1
        elif len(self.itervars) > 1:
            self.tile_row = min(self.tile_row, self.ranges[0])
            self.tile_col = min(self.tile_col, self.ranges[1])
        self.tile_size = self.tile_row * self.tile_col
        return ret

    def _codegen_kernel(self, arg_defs, kernel_name):
        code = super()._codegen_kernel(arg_defs, kernel_name)
        # Add llvm matrix intrinsics definition
        code.writeline(f'declare <{self.tile_size} x float> @llvm.matrix.column.major.load.v{self.tile_size}f32.p0f32(ptr , i64, i1, i32, i32) #2')
        if self.tile_size != self.tile_row:
            code.writeline(f'declare <{self.tile_row} x float> @llvm.matrix.column.major.load.v{self.tile_row}f32.p0f32(ptr , i64, i1, i32, i32) #2')
        code.writeline(f'declare <1 x float> @llvm.matrix.column.major.load.v1f32.p0f32(ptr , i64, i1, i32, i32) #2')
        code.writeline(f'declare void @llvm.matrix.column.major.store.v1f32.p0f32(<1 x float>, ptr , i64, i1, i32, i32) #3')
        code.writeline(f'declare <{self.tile_size} x float> @llvm.matrix.multiply.v{self.tile_size}f32.v16f32.v16f32(<16 x float>, <16 x float>, i32, i32, i32) #1')
        code.writeline(f'declare void @llvm.matrix.column.major.store.v{self.tile_size}f32.p0f32(<{self.tile_size} x float>, ptr , i64, i1, i32, i32) #3')
        if self.tile_size != self.tile_row:
            code.writeline(f'declare void @llvm.matrix.column.major.store.v{self.tile_row}f32.p0f32(<{self.tile_row} x float>, ptr , i64, i1, i32, i32) #3')
        code.writeline(f'declare <{self.tile_size} x float> @llvm.matrix.transpose.v{self.tile_size}f32.p0f32(<{self.tile_size} x float>, i32, i32) #2')
        code.writeline(f'declare float @llvm.vector.reduce.fadd.nxv2f32(float, <{self.tile_row} x float>)')
        code.writeline(f'declare float @llvm.vector.reduce.fmax.nxv2f32(<{self.tile_row} x float>)')
        code.writeline(f'declare <{self.tile_size} x float> @llvm.exp.f32(<{self.tile_size} x float>) #1')
        code.writeline(f'declare <{self.tile_size} x float> @llvm.maximum.f32(<{self.tile_size} x float>, <{self.tile_size} x float>) #1')
        return code


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    reduction_vars: Dict[str, str] = None

    # Todo. Type change for reduction
    INDEX_TYPE = "i64"
    INDEX_SIZE = 8

    def lines(self, line, stride=1):
        loop_index = self.idx
        self.stride = stride
        @contextlib.contextmanager
        def ctx():
            entry_label = f"entry{loop_index}"
            for_body_label = f"for.body{loop_index}"
            for_inc_label = f"for.inc{loop_index}"
            for_end_label = f"for.end{loop_index}"

            index = f"%index{loop_index}"
            index_next = f"%index.next{loop_index}"
            cmp_var = f"%cmp{loop_index}"

            line.writeline(f"br label %{entry_label}")
            line.writeline(f"\n{entry_label}:")
            line.writeline(f"br label %{for_body_label}")

            line.writeline(f"\n{for_body_label}:")
            line.writeline(f"{index} = phi {self.INDEX_TYPE} [ {self.start}, %{entry_label} ], [ {index_next}, %{for_inc_label} ]")
            yield
            line.writeline(f"br label %{for_inc_label}")
            line.writeline(f"\n{for_inc_label}:")
            line.writeline(f"{index_next} = add nsw {self.INDEX_TYPE} {index}, {stride}")
            line.writeline(f"{cmp_var} = icmp eq {self.INDEX_TYPE} {index_next}, {self.size}")
            line.writeline(f"br i1 {cmp_var}, label %{for_end_label}, label %{for_body_label}")

            line.writeline(f"\n{for_end_label}:")
        return ctx()

@dataclasses.dataclass
class VectorLoopLevel(LoopLevel):
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    reduction_vars: Dict[str, str] = None

    DATA_TYPE = "i32"
    DATA_SIZE = 4
    def lines(self, line, stride=1):
        loop_index = self.idx
        self.stride = stride # FIXME. vector type can't be determined in this time...
        @contextlib.contextmanager
        def ctx():
            # Label definition
            entry_label = f"vector.entry{loop_index}"
            vector_body_label = f"vector.body{loop_index}"
            ph_label = f"vector.ph{loop_index}"
            middle_label = f"middle.block{loop_index}"
            preheader_label = f"vector.for.body.preheader{loop_index}"
            for_body_label = f"vector.for.body{loop_index}"
            func_ret_label = f"for.end{loop_index}"

            # Variable definition
            entry_var0 = f"%entry_var.0"
            entry_var1 = f"%entry_var.1"
            entry_var2 = f"%entry_var.2"
            min_iter_check = f"%min.iters.check"
            ph_var0 = f"%ph_var.0"
            ph_var1 = f"%ph_var.1"
            ph_mod = f"%n_mod.vf"
            ph_vec = f"%n.vec"
            ph_stride0 = f"%ph_stride.0"
            ph_stride1 = f"%ph_stride.1"
            ph_stride2 = f"%ph_stride.2"
            min_iter_check = f"%min.iters.check{loop_index}"
            idx = f"%vector.index{loop_index}"
            idx_next = f"%vector.index.next{loop_index}"
            loop_condition = f"%vector.condition{loop_index}"
            scalar_condition = f"%scalar.condition"
            scalar_index_ph = f"%scalar.index.ph"

            line.writeline(f"br label %{entry_label}")
            line.writeline(f"\n{entry_label}:")
            line.writeline(f"{entry_var0} = tail call {self.INDEX_TYPE} @llvm.vscale.{self.INDEX_TYPE}()")
            line.writeline(f"{entry_var1} = shl nuw nsw {self.INDEX_TYPE} {entry_var0}, 2")
            line.writeline(f"{entry_var2} = tail call {self.INDEX_TYPE} @llvm.umax.{self.INDEX_TYPE}({self.INDEX_TYPE} {entry_var1}, {self.INDEX_TYPE} 16)")
            line.writeline(f"{min_iter_check} = icmp ugt {self.INDEX_TYPE} {entry_var2}, {self.size}")
            line.writeline(f"br i1 {min_iter_check}, label %{preheader_label}, label %{ph_label}")

            # Vector loop body part
            line.writeline(f"\n{ph_label}:")
            line.writeline(f"{ph_var0} = tail call {self.INDEX_TYPE} @llvm.vscale.{self.INDEX_TYPE}()")
            line.writeline(f"{ph_var1} = shl nuw nsw {self.INDEX_TYPE} {ph_var0}, 2") # FIXME. 2 is hardcoded...
            line.writeline(f"{ph_mod} = urem {self.INDEX_TYPE} {self.size}, {ph_var1}")
            line.writeline(f"{ph_vec} = sub nuw nsw {self.INDEX_TYPE} {self.size}, {ph_mod}")
            line.writeline(f"{ph_stride0} = tail call {self.DATA_TYPE} @llvm.vscale.{self.DATA_TYPE}()")
            line.writeline(f"{ph_stride1} = shl nuw nsw {self.DATA_TYPE} {ph_stride0}, 1")
            line.writeline(f"{ph_stride2} = zext {self.DATA_TYPE} {ph_stride1} to {self.INDEX_TYPE}")

            line.writeline(f"br label %{vector_body_label}")
            line.writeline(f"\n{vector_body_label}:")
            line.writeline(f"{idx} = phi {self.INDEX_TYPE} [ 0, %{ph_label} ], [ {idx_next}, %{vector_body_label} ]")
            yield

            # Increment & condition check part
            line.writeline(f"{idx_next} = add nuw {self.INDEX_TYPE} {idx}, {ph_stride2}")
            line.writeline(f"{loop_condition} = icmp eq {self.INDEX_TYPE} {idx_next}, {ph_vec}")
            line.writeline(f"br i1 {loop_condition}, label %{middle_label}, label %{vector_body_label}")

            line.writeline(f"\n{middle_label}:")
            line.writeline(f"{scalar_condition} = icmp eq {self.INDEX_TYPE} {ph_mod}, 0")
            line.writeline(f"br i1 {scalar_condition}, label %{func_ret_label}, label %{preheader_label}")

            line.writeline(f"\n{preheader_label}:")
            line.writeline(f"{scalar_index_ph} = phi {self.INDEX_TYPE} [ 0, %{entry_label} ], [ {ph_vec}, %{middle_label} ]")
        return ctx()

@dataclasses.dataclass
class MatrixLoopLevel(LoopLevel):
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    tile_row: int = 4
    reduction_vars: Dict[str, str] = None

    # Todo. Type change for reduction
    INDEX_TYPE = "i64"
    INDEX_SIZE = 8

    def lines(self, line, stride=1):
        loop_index = self.idx
        self.stride = stride * self.tile_row # FIXME: this stride is not correct (it should be stride of tile, not stride of scalar)
        @contextlib.contextmanager
        def ctx():
            entry_label = f"entry{loop_index}"
            for_body_label = f"for.body{loop_index}"
            for_inc_label = f"for.inc{loop_index}"
            for_end_label = f"for.end{loop_index}"

            index = f"%index{loop_index}"
            index_next = f"%index.next{loop_index}"
            cmp_var = f"%cmp{loop_index}"

            line.writeline(f"br label %{entry_label}")
            line.writeline(f"\n{entry_label}:")
            line.writeline(f"br label %{for_body_label}")

            line.writeline(f"\n{for_body_label}:")
            line.writeline(f"{index} = phi {self.INDEX_TYPE} [ {self.start}, %{entry_label} ], [ {index_next}, %{for_inc_label} ]")
            yield
            line.writeline(f"br label %{for_inc_label}")
            line.writeline(f"\n{for_inc_label}:")
            line.writeline(f"{index_next} = add nsw {self.INDEX_TYPE} {index}, {self.tile_row}")
            line.writeline(f"{cmp_var} = icmp eq {self.INDEX_TYPE} {index_next}, {self.size}")
            line.writeline(f"br i1 {cmp_var}, label %{for_end_label}, label %{for_body_label}")

            line.writeline(f"\n{for_end_label}:")
        return ctx()

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

    def codegen(self, code, stack):
        size_list = []
        stride_list = []
        loop_info = {}
        for loop in self.loops:
            stride_list.append(loop.size)
        stride_list.append(1)

        var = 1
        for sz in size_list[::-1]:
            var = var * sz
            stride_list.append(var)
        stride_list = stride_list[::-1]

        for loop, stride in zip(self.loops, stride_list):
            stack.enter_context(loop.lines(code, stride=stride))
            loop_info[str(loop.var)] = [loop.start, loop.size, loop.stride]
        return loop_info

class LLVMScheduling(BaseScheduling):
    count = 0
    target_kernel = LLVMKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return False

    def can_fuse_horizontal(self, node1, node2):
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = self.target_kernel()

        kernel_name = f"extension_kernel_{self.count}"
        self.count += 1
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name)
        self.define_kernel(src_code, kernel_name)
        ex_kernel.call_kernel(kernel_name)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()

    def define_function(self, kernel):
        code = kernel.def_function()
        if code is not None:
            wrapper = V.graph.wrapper_code
            wrapper.header.writeline(code)

    def define_kernel(self, src_code, kernel_name):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            wrapper.src_to_kernel[src_code] = kernel_name

            codecache_def = IndentedBuffer()
            codecache_def.writeline("custom_async_compile.llvm('''")
            codecache_def.splice(src_code)
            codecache_def.writeline("''', ")
            codecache_def.writeline("loop_info=loop_info,")
            codecache_def.writeline("load_tile_info=load_tile_info,")
            codecache_def.writeline("store_tile_info=store_tile_info,")
            codecache_def.writeline("arg_attributes=arg_attributes)")

            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name

class VectorizedLLVMScheduling(LLVMScheduling):
    target_kernel = VectorizedLLVMKernel

class MatrixLLVMScheduling(LLVMScheduling):
    target_kernel = MatrixLLVMKernel
    def codegen_template(self, template_node, epilogue_nodes):
        _, (numel, rnumel) = template_node.group

        template_buffer = template_node.node
        kernel, render = template_buffer.make_kernel_render(template_buffer, epilogue_nodes=epilogue_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel.meta_kernel()
            kernel_name = self.define_kernel(src_code, kernel.kernel_name)
            self.define_function(kernel)
        kernel.call_kernel(kernel_name)