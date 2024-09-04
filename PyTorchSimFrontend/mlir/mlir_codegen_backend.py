import dataclasses
import contextlib
import sympy
import itertools
import re
import os
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
from torch._inductor.codecache import write_atomic, write
import extension_codecache

from . import mlir_common
from . import mlir_lowering

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

def reduction_combine_vec(reduction_type, vector_value, tile_size=64):
    if reduction_type == "sum":
        return f"vector.reduction <add>, %{vector_value} : vector<{tile_size}xf32> into f32"
    if reduction_type == "prod":
        return f"vector.reduction <mul>, %{vector_value} : vector<{tile_size}xf32> into f32"
    if reduction_type in ("min", "max"):
        raise NotImplementedError()
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
    @staticmethod
    def add(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.addf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def sub(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.subf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def mul(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.mulf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def div(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.divf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def truediv(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.divf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def constant(value, dtype, tile_size=16):
        return f'arith.constant {value} : {mlir_common.DTYPE_TO_MLIR[dtype]}'

    @staticmethod
    def exp(operand, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'math.exp %{operand} : {shape}'

RTYPE_TO_MLIR = {
    "sum": "add",
    "prod": "mul",
}

class MLIRKernel(mlir_common.BaseMLIRKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, args=None):
        super().__init__(mlir_common.MLIRKernelArgs())
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.global_vars = IndentedBuffer()
        self.header = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.iterator_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="iter")
        self.init_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init")
        self.init_vec_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init_vec")
        self.map_cse = common.CSE("#", self.suffix, name_prefix="map")
        self.spad_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="spad")
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.tiling_indices = [0, 1]

    def parse_indices(self, expr):
        if len(expr.args) == 0:
            return expr
        pattern = r'index\d+'
        indices = re.findall(pattern, str(expr))
        args = ", ".join(map(str, indices))
        self.map_cse.generate(self.global_vars, f"affine_map<({args}) -> ({expr})>")
        args = ", ".join([f"%{i}" for i in indices])
        index = self.cse.generate(self.loads, f"affine.apply #map0({args})")
        return index

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
        _, _, _, self.buffer_types = self.args.mlir_argdefs()
        with self as kernel:
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                node.run(vars, reduction_vars)

        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()

        write_path = extension_codecache.get_write_path(src_code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        write_path = os.path.join(write_path, "global_var.h")
        if not os.path.exists(write_path):
            write_atomic(write_path, self.header.getvalue())
        return src_code

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        indices = self.parse_indices(index)
        prefix = "" if index.is_number else "%"
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        tile_size = min(self.tile_size, self.buffer_types[name][1])
        spad_size = tile_size * self.vector_lane
        self.header.writeline(f"{mlir_common.DTYPE_TO_C[dtype]} {name}_spad[{spad_size}] __attribute__ ((section(\".spad\")));")
        self.spad_cse.generate(self.global_vars, f"memref.global @{name}_spad : memref<{spad_size}x{type_name}, 1>", assignment = False)
        buffer = self.cse.generate(self.loads, f"memref.get_global @{name}_spad : memref<{spad_size}x{type_name}, 1>")
        code = f"affine.dma_start %{var}[{prefix}{indices}], %{buffer}[0], %{name}_tag[0], %c{spad_size}, %c1, %c{spad_size} : memref<{self.buffer_types[name][1]}x{type_name}>, memref<{spad_size}x{type_name}, 1>, memref<1xi32>"
        self.cse.generate(self.loads, code, assignment = False)
        operation = "affine.vector_load" if tile_size > 1 else "affine.load"
        shape = f", vector<{tile_size}x{type_name}>" if tile_size > 1 else ""
        line = f"{operation} %{buffer}[0] : memref<{spad_size}x{type_name}, 1>{shape}"
        out = self.cse.generate(self.loads, line)
        self.tile_info[out] = tile_size, dtype
        return out

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        indices = self.parse_indices(index)
        prefix = "" if index.is_number else "%"
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        tile_size = min(self.tile_size, self.buffer_types[name][1])
        spad_size = tile_size * self.vector_lane
        self.header.writeline(f"{mlir_common.DTYPE_TO_C[dtype]} {name}_spad[{spad_size}] __attribute__ ((section(\".spad\")));")
        self.spad_cse.generate(self.global_vars, f"memref.global @{name}_spad : memref<{spad_size}x{type_name}, 1>", assignment = False)
        buffer = self.cse.generate(self.stores, f"memref.get_global @{name}_spad : memref<{spad_size}x{type_name}, 1>")
        operation = "affine.vector_store" if tile_size > 1 else "affine.store"
        shape = f", vector<{tile_size}x{type_name}>" if tile_size > 1 else ""
        line = f"{operation} %{value}, %{buffer}[0] : memref<{spad_size}x{type_name}, 1>{shape}"
        self.cse.generate(self.stores, line, assignment = False)
        code = f"affine.dma_start %{buffer}[0], %{var}[{prefix}{indices}], %{name}_tag[0], %c{spad_size}, %c1, %c{spad_size} : memref<{spad_size}x{type_name}, 1>, memref<{self.buffer_types[name][1]}x{type_name}>, memref<1xi32>"
        self.cse.generate(self.stores, code, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
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
            type_name = mlir_common.DTYPE_TO_MLIR[dtype]
            self.reduction_vars[acc] = (reduction_type, iterator, init, type_name)
            self.reduction_prefix.writeline(f"%{init} = arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
            if self.tiling_idx >= self.reduction_depth: # horizontal reduction
                out = self.cse.generate(self.compute, reduction_combine_vec(reduction_type, value, tile_size=self.tile_size))
                out = self.cse.generate(self.compute, f"arith.{RTYPE_TO_MLIR[reduction_type]}f %{iterator}, %{out} : {type_name}")
                self.cse.generate(self.stores, f"affine.yield %{out} : {type_name}", assignment = False)
            else:
                init_vec = self.init_vec_cse.generate(
                    self.loads, f"reduction {reduction_key}", write=False
                )
                self.reduction_prefix.writeline(f"%{init_vec} = vector.broadcast %{init} : {type_name} to vector<{self.tile_size}x{type_name}>")
                out = self.cse.generate(self.compute, reduction_combine(reduction_type, iterator, value, tile_size=self.tile_size))
                self.cse.generate(self.compute, f"affine.yield %{out} : vector<{self.tile_size}x{type_name}>", assignment = False)
                self.reduction_vars[acc] = (reduction_type, iterator, init_vec, f"vector<{self.tile_size}x{type_name}>")
            self.reduction_cse.reduction_cache[reduction_key] = acc
            self.iterator_cse.reduction_cache[reduction_key] = iterator
            self.init_cse.reduction_cache[reduction_key] = init
        return acc

    def store_reduction(self, name, index, value):
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        index = self.rename_indexing(index)
        indices = self.parse_indices(index)
        prefix = "" if index.is_number else "%"
        if self.tiling_idx >= self.reduction_depth: # horizontal reduction
            self.cse.generate(self.reductions_suffix, f"affine.store %{value}, %{var}[{prefix}{indices}] : memref<{self.buffer_types[name][1]}x{type_name}>", assignment = False)
        else:
            self.cse.generate(self.reductions_suffix, f"affine.vector_store %{value}, %{var}[{prefix}{indices}] : memref<{self.buffer_types[name][1]}x{type_name}>, vector<{self.tile_size}x{type_name}>", assignment = False)

    def codegen_init(self):
        code = IndentedBuffer()
        _, _, arg_attributes, _ = self.args.mlir_argdefs()
        tiles = set()
        for name, (_, dtype, size) in arg_attributes.items():
            tile_size = min(self.tile_size, size)
            type_name = mlir_common.DTYPE_TO_MLIR[dtype]
            code.writeline(f"%{name}_tag = memref.alloc() : memref<1xi32>")
            tiles.add(tile_size)
        for tile_size in tiles:
            spad_size = tile_size * self.vector_lane
            code.writeline(f"%c{spad_size} = arith.constant {spad_size} : index")
        code.writeline(f"%c1 = arith.constant 1 : index") # for stride TODO: matrix dma stride should be col_size
        return code

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop body part
        loops = [LoopLevel(var, size, idx, tile_size=1, vector_lane=self.vector_lane) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
        if len(loops) > 0:
            loops[self.tiling_idx].tile_size = self.tile_size #innermost vector tile
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            for loop in loops.loops:
                loop_lines = loop.lines()
                if loop_lines is None:
                    return
                code.writelines(loop_lines)
                stack.enter_context(code.indent())
            with contextlib.ExitStack() as stack_outer:
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    for reduction in reductions.loops:
                        reduction_lines = reduction.lines()
                        if reduction_lines is None:
                            return
                        code.writelines(reduction_lines)
                        stack.enter_context(code.indent())
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"return")
        return code

    def codegen_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        arg_defs, _, _, _ = self.args.mlir_argdefs()
        code = self._codegen_kernel(arg_defs, kernel_name)
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes, _ = self.args.mlir_argdefs()
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
        _, call_args, _, _ = self.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def _codegen_kernel(self, arg_defs, kernel_name):
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        code.splice(self.global_vars)
        #TODO:. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_init())
            code.splice(self.codegen_loops())
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
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    tile_size: int = 4
    vector_lane: int = 128
    reduction_vars: Dict[str, str] = None

    def lines(self):
        loop_index = self.idx
        if self.reduction_vars:
            acc = ', '.join([acc.name for acc in self.reduction_vars.keys()])
            args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
            dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
            line = f"%{acc} = affine.for %index{loop_index} = {self.start} to {self.size} step {self.tile_size} iter_args({args}) -> ({dtype})"
        else:
            line = f"affine.for %index{loop_index} = {self.start} to {self.size} step {self.tile_size * self.vector_lane}"

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

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
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
            codecache_def.writeline("custom_async_compile.mlir('''")
            codecache_def.splice(src_code)
            codecache_def.writeline("''', ")
            codecache_def.writeline("loop_info=loop_info,")
            codecache_def.writeline("load_tile_info=load_tile_info,")
            codecache_def.writeline("store_tile_info=store_tile_info,")
            codecache_def.writeline("arg_attributes=arg_attributes)")

            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name

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