import contextlib
import sympy
import time
import re
import os
from functools import reduce
from operator import mul
import torch
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PyTorchSimFrontend import extension_config
from torch._dynamo.testing import rand_strided
from torch._inductor.autotune_process import TensorMeta
from torch._dynamo.utils import dynamo_timed
from torch._inductor.codegen import cpp, wrapper, common, memory_planning
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.utils import (
    IndentedBuffer,
    is_welford_reduction,
    sympy_product
)
from torch.utils._sympy.functions import ModularIndexing, FloorDiv
from PyTorchSimFrontend import extension_codecache
from PyTorchSimFrontend import extension_functional_verify as _func_verify
from . import mlir_common
from .mlir_common import LoopLevel, LoopNest
from .mlir_ops import ExtensionOverrides
from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest

# Configure logger for mlir_codegen_backend module
logger = extension_config.setup_logger()

from Simulator.simulator import ProgressBar

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
        return "-inf"
    if reduction_type in {"min", "argmin"}:
        return "inf"
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
        return ops.logical_or(vector_value, init_value)
    raise AssertionError(reduction_type)

class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self):
        super().__init__()

    @classmethod
    def create(
        cls,
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[wrapper.PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None and parent_wrapper is not None
            return wrapper.SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return cls()

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
                from torch._inductor.async_compile import AsyncCompile

                from torch import device, empty, empty_strided
                from {extension_codecache.__name__} import CustomAsyncCompile
                from PyTorchSimFrontend.extension_config import CONFIG_SRAM_BUFFER_PLAN, setup_logger
                from Simulator.simulator import TOGSimulator
                from PyTorchSimFrontend.extension_op import sparse_mm_dummy_stonne_outer
                from PyTorchSimFrontend import extension_functional_verify as _fverify
                from torch._inductor.select_algorithm import extern_kernels

                # Configure logger for generated wrapper code
                _logger = setup_logger("PyTorchSimFrontend.mlir.generated_wrapper")

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                custom_async_compile = CustomAsyncCompile()
                async_compile = AsyncCompile()
                os.environ["TORCHSIM_LAST_COMPILED_MODULE"] = __file__
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )
        self.header.splice(
            f"""
            def sram_plan_prefix(buffer_name, buffer):
                if CONFIG_SRAM_BUFFER_PLAN and (buffer_name not in CONFIG_SRAM_BUFFER_PLAN):
                    return
                buffer_size = buffer.untyped_storage().size()
                start = buffer.data_ptr()
                end = start + buffer_size
                # print(f'Alloc {{buffer_name}}(0x{{start:x}} ~ 0x{{end:x}})')
                TOGSimulator.sram_alloc(buffer_name, [start, end])

            def sram_plan_postfix(buffer_name, buffer):
                if CONFIG_SRAM_BUFFER_PLAN and (buffer_name not in CONFIG_SRAM_BUFFER_PLAN):
                    return
                buffer_size = buffer.untyped_storage().size()
                start = buffer.data_ptr()
                end = start + buffer_size
                # print(f'Dealloc {{buffer_name}}(0x{{start:x}} ~ 0x{{end:x}})')
                TOGSimulator.sram_dealloc(buffer_name, [start, end])

            def host2device_memcopy(buffer):
                pass

            def device2host_memcpy(buffer):
                pass
            """
        )

    def write_prefix(self):
        self.write_async_compile_wait()
        self.prefix.splice(
            """
            def call(args):
            """
        )
        with self.prefix.indent():
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            # Per-kernel functional verify: register the runnable aten graph and
            # emit a CPU golden build at the top of call(), passing graph inputs.
            if _func_verify.enabled():
                gm = getattr(V.graph, "module", None)
                if gm is not None:
                    gid = _func_verify.register_graph(gm)
                    in_names = list(V.graph.graph_inputs.keys())
                    self.prefix.writeline(
                        f"_fverify.verify_init({gid}, [{', '.join(in_names)}])")

            self.codegen_inputs()
            self.codegen_input_size_asserts()
            self.codegen_sram_plan_prefix()

    def codegen_sram_plan_prefix(self):
        for name, buf in V.graph.graph_inputs.items():
            if buf is None:
                continue
            if isinstance(buf, sympy.Expr):
                continue
            if sympy_product(buf.get_size()) == 0:
                continue
            self.prefix.writeline(f"sram_plan_prefix('{name}', {name})")

    def codegen_sram_plan_postfix(self, outputs):
        for name in outputs:
            if name is None or name == "None":
                continue
            self.wrapper_call.writeline(f"sram_plan_postfix('{name}', {name})")

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        graph_name="",
        original_fxnode_name=None,
    ):
        device = device or V.graph.get_current_device_or_throw()
        self.writeline(self.wrap_kernel_call(kernel_name, call_args))
        return

    def generate(self, is_inference):
        result = IndentedBuffer()
        # result.splice(self.header)

        self._fverify_seen = set()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            self.memory_plan_reuse()
            with self.set_writeline(self.wrapper_call.writeline):
                for line in self.lines:
                    # Add buffer plan hook for dealloc
                    if isinstance(line, memory_planning.DeallocFromPoolLine):
                        self.wrapper_call.writeline(f"sram_plan_postfix('{line.node.get_name()}', {line.node.get_name()})")
                    elif isinstance(line, str) and "del" in line:
                        name = line.split(" ")[1]
                        self.wrapper_call.writeline(f"sram_plan_postfix('{name}', {name})")

                    if isinstance(line, wrapper.MemoryPlanningLine):
                        line.codegen(self.wrapper_call)
                    elif isinstance(line, wrapper.KernelCallLine):
                        self.wrapper_call.writeline(self.wrap_kernel_call(line.kernel_name, line.call_args))
                        if _func_verify.enabled():
                            self._fverify_emit_checks(line.call_args)
                    else:
                        if isinstance(line, wrapper.WrapperLine):
                            line.codegen(self.wrapper_call)
                        else:
                            self.wrapper_call.writeline(line)
                    # Add buffer plan hook for alloc
                    if isinstance(line, memory_planning.AllocFromPoolLine) or isinstance(line, wrapper.AllocateLine):
                        self.wrapper_call.writeline(f"sram_plan_prefix('{line.node.get_name()}', {line.node.get_name()})")
            output_refs = self.get_output_refs()
            self.codegen_sram_plan_postfix(output_refs)
            self.mark_output_type()
            self.generate_return(output_refs)

        # self.append_precomputed_sizes_to_prefix() # FIXME: Need to replace append_precomputed_sizes_to_prefix()
        result.splice(self.header)

        self.finalize_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)
        self.add_benchmark_harness(result)
        return (
            result.getvaluewithlinemap(),
            self.kernel_declarations.getvaluewithlinemap(),
        )

    def _fverify_emit_checks(self, call_args):
        """Emit per-kernel CPU verify calls for this kernel's output buffers.

        A buffer's value is produced by the first kernel that names it (producer
        precedes consumers in topo order), so we check each bare-identifier buffer
        arg the first time it is seen -- that occurrence is its output. The buffer
        is mapped to its originating fx node (op) so the runtime check can compare
        against the CPU golden keyed by that node.
        """
        for a in call_args:
            if not isinstance(a, str):
                continue
            name = a.strip()
            if not name.isidentifier() or name in self._fverify_seen:
                continue
            self._fverify_seen.add(name)
            if name in V.graph.graph_inputs:
                continue  # placeholders: golden == input, nothing to verify
            try:
                buf = V.graph.get_buffer(name)
            except Exception:
                buf = None
            if buf is None:
                continue
            origin = getattr(buf, "origin_node", None)
            if origin is None:
                continue
            op = str(getattr(origin, "target", "?"))
            self.wrapper_call.writeline(
                f'_fverify.verify_check({name}, "{name}", "{origin.name}", "{op}")')

    def memory_plan(self):
        self.lines = memory_planning.MemoryPlanner(self).plan(self.lines)

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


class Step:
    """One load->compute->store unit of the kernel body (see codegen_loops).

    Bundles the DMA, mask, index and compute buffers so the body can be an
    ordered list of steps; the formerly ad-hoc mask/index buffers are just
    fields here.
    """
    __slots__ = ("applys", "dma_loads",
                 "loads", "compute", "stores", "dma_stores")

    def __init__(self, **buffers):
        for name, buf in buffers.items():
            setattr(self, name, buf)


class MLIRKernel(mlir_common.BaseMLIRKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, kernel_group, reason=None):
        super().__init__(kernel_group, reason=reason)
        self.const_buffer = IndentedBuffer()
        self.alloc_buffer = IndentedBuffer()
        self.spad_buffer = IndentedBuffer()
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        # Kernel body = ordered load->compute->store steps; step 0 keeps the base
        # loads/compute/stores (the CSE target default captured self.compute at init).
        step0 = Step(
            applys=IndentedBuffer(),
            dma_loads=IndentedBuffer(), dma_stores=IndentedBuffer(),
            loads=self.loads, compute=self.compute, stores=self.stores,
        )
        self.steps = [step0]
        self._bind_step(step0)
        self.global_vars = IndentedBuffer()
        self.header = IndentedBuffer()
        self.gem5_header = IndentedBuffer()
        self.header.writeline("#include <unistd.h>")
        self.header.writeline("#include <stdlib.h>")
        self.header.writeline("#include <stdio.h>")
        self.header.writeline("void* __wrap_malloc(size_t size) {")  # Align to 512 bytes
        self.header.writeline("    size_t aligned = (size + 511UL) & ~511UL;")
        self.header.writeline("    void *p = sbrk(aligned);")
        #self.header.writeline('    fprintf(stderr, "[SPIKE][__wrap_malloc] addr=%p size=%zu (req=%zu)\\n", p, aligned, size);')
        self.header.writeline("    return p;")
        self.header.writeline("}")
        self.header.writeline("void __wrap_free(void *ptr) { return; }")
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.spad_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="spad")
        self.apply_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="apply")
        self.mask_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="mask")
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
        self.spadbuf_counter = 0
        self.dma_read_counter = 1
        self.dma_write_counter = 1
        self.dma_tag_id = 0
        self.affine_yield = {}
        self.welford_reduce_out = None
        self.reduce_iterator = {}
        self.spad_buffer_dict = dict()
        self.indirect_symbols = set()  # CSE-var names bound as indirect indices
        self.base_vector_initialized = False
        self.loop_size = None

    def reset(self, reason):
        save = self.exit_stack, self._nested_context_depth
        self.__init__(self.kernel_group, reason=reason)
        self.exit_stack, self._nested_context_depth = save

    @staticmethod
    def _origin_is_exp(o):
        """True iff the FX origin node is an aten.exp op -- matched by the op target, not a
        name substring (so it does not fire on expand / expm1 / experimental)."""
        t = getattr(o, "target", None)
        pkt = getattr(t, "_overloadpacket", None)
        name = getattr(pkt, "__name__", None) or getattr(t, "__name__", None) or ""
        return name == "exp"

    # padding type 0: zero-padding 1: negative-padding(-inf) ...
    def get_padding_type(self):
        ops = self.current_node.node.origins
        if self.current_node.is_reduction():
            for op in ops:
                if self._origin_is_exp(op): # exponential reduciton case
                    return 1
        # for op in ops: # TODO: padding has some problem in the case of max_pool
        #     if "max_pool" in op.args[0].name:
        #         return 1
        return 0

    def convert_index(self, expr):
        if len(expr.free_symbols) != 1:
            raise NotImplementedError("Not supporting this view operation...!")

        if expr.is_symbol:
            return expr

        expr_str = str(expr)
        if isinstance(expr, ModularIndexing):
            dim = list(expr.args[0].free_symbols)[0]
            replace_str = f"({expr.args[0]} floordiv {expr.args[1]}) mod {expr.args[2]}"
            expr_str = re.sub(r"ModularIndexing\([^)]*\)", replace_str, expr_str)
        elif "//" in expr_str:
            expr_str = expr_str.replace("//", " floordiv ")
        else:
            raise NotImplementedError("What is this case?")

        first_arg = expr.args[0]
        if len(first_arg.free_symbols) != 1:
            raise NotImplementedError("What is this case?")

        # Create affine.apply operation
        indices = [list(first_arg.free_symbols)[0]]
        with self.override_buffer_cse(buffer=self.global_vars, cse=self.map_cse):
            map_var = ops.affine_map(indices, expr_str)
        index = ops.affine_apply(map_var, indices)
        return index

    def _convert_sympy_to_mlir_expr(self, expr, sorted_args):
        """
        Convert sympy expression to MLIR affine map expression by replacing index variables.
        """
        indices = []

        for arg in sorted_args:
            if arg.is_Mul and arg.args[0].is_number:
                target_arg = arg.args[1]
            elif not arg.is_number:
                target_arg = arg
            else:
                continue
            new_arg = sympy.Symbol(str(self.convert_index(target_arg)))
            expr = expr.replace(target_arg, new_arg)
            indices.append(str(new_arg))

        # Convert ModularIndexing and FloorDiv to sympy expressions
        # ModularIndexing(x, y, z) means (x // y) % z -> Mod(FloorDiv(x, y), z)
        # FloorDiv(x, y) means x // y -> will be converted to floordiv in string representation
        # Use preorder_traversal to find all instances
        replacements = {}
        for sub in sympy.preorder_traversal(expr):
            if isinstance(sub, ModularIndexing):
                # Convert ModularIndexing to Mod(FloorDiv(...), ...)
                if sub.args[1] != 1:
                    floor_div = FloorDiv(sub.args[0], sub.args[1])
                else:
                    floor_div = sub.args[0]
                mod_expr = sympy.Mod(floor_div, sub.args[2])
                replacements[sub] = mod_expr
            elif isinstance(sub, FloorDiv):
                # Keep FloorDiv as is, will be handled in custom string conversion
                # We need to mark it for special handling
                pass

        # Apply replacements
        for old_expr, new_expr in replacements.items():
            expr = expr.subs(old_expr, new_expr)

        # Custom string conversion for MLIR affine expressions
        def mlir_str(expr):
            """Convert sympy expression to MLIR affine expression string"""
            if isinstance(expr, FloorDiv):
                return f"({mlir_str(expr.args[0])} floordiv {mlir_str(expr.args[1])})"
            elif isinstance(expr, sympy.Mod):
                return f"({mlir_str(expr.args[0])} mod {mlir_str(expr.args[1])})"
            elif isinstance(expr, sympy.Add):
                terms = [mlir_str(term) for term in expr.args]
                return " + ".join(terms)
            elif isinstance(expr, sympy.Mul):
                factors = [mlir_str(factor) for factor in expr.args]
                return " * ".join(factors)
            elif isinstance(expr, sympy.Symbol):
                return str(expr)
            elif expr.is_number:
                return str(expr)
            else:
                # Fallback to string representation
                return str(expr)

        expr_str = mlir_str(expr)
        return expr_str, indices

    def parse_indices(self, expr, comments="", indices=None, indirect_dims=[]) -> common.CSEVariable:
        # Constant case
        if expr.is_number and len(indirect_dims) == 0:
            return self.get_const_cse(int(expr))

        # Identity case
        if len(expr.args) == 0 and len(indirect_dims) == 0:
            return expr

        if len(expr.args) == 0:
            args = [expr]
        else:
            args = list(expr.args)
        # Sort index variable.. ex) (%index1, %index0)
        args_dict = {term: list(term.free_symbols)[0] for term in args if term.free_symbols}
        sorted_args = sorted(args_dict.keys(), key=lambda term: str(args_dict[term]))

        # Convert sympy expression to affine map expression
        expr_str, indices = self._convert_sympy_to_mlir_expr(expr, sorted_args)
        indirect_args = [f"%{i}" for i in indirect_dims]
        # Create affine.apply operation
        with self.override_buffer_cse(buffer=self.global_vars, cse=self.map_cse):
            map_var = ops.affine_map(indices, expr_str, symbol_names=indirect_dims)

        index = ops.affine_apply(map_var, indices, indirect_dims=indirect_args, comment=comments)
        return index

    def parse_index_list(self, expr_list:list, offset=sympy.Number(0)) -> common.CSEVariable:
        """ Need to override buffer and cse to use this function. """
        expr_list = [arg for arg in expr_list]
        dim_list = [f"d{i}" for i in range(len(expr_list))]

        if len(expr_list) == 1 and expr_list[0].is_number:
            # Constant case
            return self.get_const_cse(int(expr_list[0] + offset))
        elif len(expr_list) == 1 and expr_list[0].is_symbol and int(offset) == 0:
            # Identity case
            return expr_list[0]

        indices = []
        new_expr_list = [0] * len(expr_list)
        for idx, arg in enumerate(expr_list):
            if arg.is_Mul and arg.args[0].is_number:
                new_arg = sympy.Symbol(str(self.convert_index(arg.args[1])))
                new_expr_list[idx] = arg.subs(arg.args[1], dim_list[idx])
                indices.append(str(new_arg))
            elif not arg.is_number:
                try:
                    new_arg = sympy.Symbol(str(self.convert_index(arg)))
                #not implemented case
                except NotImplementedError:
                    print(f"Not implemented case: {arg}")
                    raise NotImplementedError(f"Not implemented case: {arg}")
                new_expr_list[idx] = new_arg.subs(new_arg, dim_list[idx])
                indices.append(str(new_arg))
            else:
                const_var = self.get_const_cse(int(arg))
                new_arg = sympy.Symbol(f"{const_var}")
                new_expr_list[idx] = arg
                indices.append(str(new_arg))

        # Extract index var
        # Create affine.apply operation
        expr_str = str(sum(new_expr_list) + offset)
        with self.override_buffer_cse(buffer=self.global_vars, cse=self.map_cse):
            map_var = ops.affine_map(dim_list, expr_str)
        index = ops.affine_apply(map_var, indices)
        return index

    def load(self, name: str, index: sympy.Expr):
        index, offset_desc = self.convert_indirect_indexing(index)
        padding = self.get_padding_type()

        # Extract dram info
        dram_var = self.kernel_group.args.input(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        # Extract sram info
        local_tile_desc, index_var, dram_stride, local_dims = self.get_dma_info(name, index)
        vlane_split_axis = local_tile_desc.vmap.vlane_split_axis
        vlane_stride = local_tile_desc.vmap.vlane_stride
        tile_numel_per_lane = local_tile_desc.get_numel_per_lane()
        tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = local_tile_desc.get_tile_stride()
        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        # Define scratch pad buffer
        sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, local_tile_desc, index)
        compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])

        masked_bounds = self._masked_bounds(name, index, dram_stride, local_tile_desc, is_load=True, buffer=self.dma_loads, local_dims=local_dims)
        code = self.emit_transfer("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                  dram_shape, tile_shape, dram_stride, tile_stride, int(padding), offset=offset_desc,
                                  masked_bounds=masked_bounds, masked_fill=self._masked_fill_bits(dtype, index))
        self.cse.generate(self.dma_loads, code, assignment = False) # FIXME: assignment = False does not support caching

        with self.override_buffer_cse(buffer=self.loads):
            out = ops._load(compute_vec_size, mlir_dtype, sram_var, compute_index_var, tile_shape)
        self.spad_buffer_dict[str(out)] = [sram_var, local_tile_desc.get_tile_size(), tile_numel_per_lane, sram_index_var, tile_shape, vshape]
        return out

    def store(self, name: str, index: sympy.Expr, value, mode=None, *args, **kwargs):
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        offset_desc = None

        # Handle scatter store
        accumulate = False
        if self._has_indirect(index):
            # Convert the output buffer type to the inplace buffer
            arg_name =  V.graph.scheduler.mutation_real_name.get(name, name)
            if arg_name not in self.kernel_group.args.inplace_buffers:
                self.kernel_group.args.make_inplace(arg_name, arg_name)

            # index_add: let the MVOUT do out[idx] += val. The DMA processes positions
            # sequentially, so duplicate indices accumulate correctly regardless of tile
            # size -- unlike the compute gather-add-overwrite, which loses duplicates that
            # land in the same (non-dividing) tile.
            accumulate = (mode == "atomic_add")
            index, offset_desc = self.convert_indirect_indexing(index)
        dram_var = self.kernel_group.args.output(name)

        # Prepare dma instruction
        local_tile_desc, index_var, dram_stride, local_dims = self.get_dma_info(name, index)
        vlane_split_axis = local_tile_desc.vmap.vlane_split_axis
        vlane_stride = local_tile_desc.vmap.vlane_stride

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
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, local_tile_desc, index)
            compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])
            # Generate vector store instruction
            _, operand_type = self.var_info[value]
            if mlir_dtype != operand_type:
                value = ops.to_dtype(value, mlir_dtype)

            if compute_vec_size < self.var_info[value][0]:
                with self.override_buffer_cse(buffer=self.stores):
                    value = ops.extract_strided_slice(value, compute_vec_size)

            with self.override_buffer_cse(buffer=self.stores):
                ops._store(value, sram_var, compute_index_var, tile_shape, buffer_name=name)
        else:
            sram_var = self.spad_buffer_dict[str(value)][0]
            sram_index_var = self.spad_buffer_dict[str(value)][3]

        # Generate DMA instruction
        masked_bounds = self._masked_bounds(name, index, dram_stride, local_tile_desc, is_load=False, buffer=self.dma_stores, local_dims=local_dims)
        code = self.emit_transfer("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                  dram_shape, tile_shape, dram_stride, tile_stride, 0, offset=offset_desc, masked_bounds=masked_bounds,
                                  accumulate=accumulate, acc_float=dtype.is_floating_point)
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
                if self.welford_reduce_out is not None:
                    return self.welford_reduce_out
                else:
                    self.welford_reduce_out = (sum, sqr_sum, None)
                    return sum, sqr_sum, None

        # Prepare reduction loop
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        vec_len = self.kernel_group.tile_desc.get_compute_vec_size()
        reduced_shape = self.kernel_group.tile_desc.get_mlir_vshape(type_name)



        # Prepare reduction init
        with self.override_buffer_cse(cse=self.const_cse, buffer=self.const_buffer):
            init = self.get_const_cse(reduction_init(reduction_type, dtype), type_name)
            init_vec = init if vec_len == 1 else ops.broadcast(init, vec_len)

        acc_var_list = []
        iter_var_list = []
        for reduction_depth in range(self.get_nr_rdim()):
            # Create reduction key
            reduction_key = src_dtype, reduction_type, value, reduction_depth
            acc_init_var = init_vec if reduction_depth == 0 else iter_var_list[-1]

            acc = self.reduction_cse.generate(self.loads, f"reduction {reduction_key}", write=False)
            iterator = self.iterator_cse.generate(self.loads, f"reduction {reduction_key}", write=False)
            acc_var_list.append(acc)
            iter_var_list.append(iterator)

            # Register reduction info
            self.reduction_vars[acc] = (reduction_type, iterator, acc_init_var, reduced_shape, reduction_depth)
            self.reduction_cse.reduction_cache[reduction_key] = acc

        # Reduction body prepare
        # Note: reduction body is inner most loop body. So it doesn't need reduction depth.
        body_key = src_dtype, reduction_type, value
        body_acc = self.reduction_cse.generate(self.compute, f"reduction {body_key}body_acc", write=False)
        body_iter_arg = self.iterator_cse.generate(self.compute, f"reduction {body_key}body_iter_arg", write=False)
        self.register_var_info(body_iter_arg, [vec_len, type_name])
        acc_var_list.append(body_acc)

        # Reduction body codegen
        _, mask_var = self.get_mask()
        if mask_var is not None:
            value = ops.where(mask_var, value, init_vec)

        result = reduction_partial_combine_vec(reduction_type, value, body_iter_arg)
        result = ops.to_dtype(result, type_name)

        self.compute_body_loop.reduction_vars[body_acc] = (reduction_type, body_iter_arg, iter_var_list[-1], reduced_shape)
        self.compute_body_loop.affine_yield[result] = reduced_shape
        # Register affine yield var
        for reduction_depth, acc in enumerate(acc_var_list[1:]):
            self.affine_yield[acc] = reduced_shape, reduction_depth

        # Final reduction
        reduction_size = self.kernel_group.tile_desc.get_numel_per_lane() // self.kernel_group.tile_desc.get_reduction_numel()
        acc = acc_var_list[0] # Set outermost acc var
        self.register_var_info(acc, [reduction_size, type_name])
        assert(vec_len % reduction_size==0)

        # Prepare init value
        init = self.get_const_cse(reduction_init(reduction_type, dtype), type_name)
        if reduction_size != 1:
            with self.override_buffer_cse(buffer=self.reductions_suffix):
                init = ops.broadcast(init, reduction_size)

        # Final reduction codegen
        with self.override_buffer_cse(buffer=self.reductions_suffix):
            if vec_len > reduction_size:
                acc = ops.multi_reduction(acc, init, vec_len, reduction_size, reduced_shape, reduction_type, type_name)
        return acc

    def store_reduction(self, name, index, value):
        # Store reduction can't share cached value stored in cse,
        # since it is not innermost loop body.
        dram_var = self.kernel_group.args.output(name)
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        with self.override_buffer_cse(cse=self.reduction_cse):
            # Tile is always reuduced in inner loop
            local_tile_desc, index_var, dram_stride, _ = self.get_dma_info(name, index, broadcast=False, store_reduction=True, buffer=self.reductions_suffix)
            vlane_split_axis = local_tile_desc.vmap.vlane_split_axis
            vlane_stride = local_tile_desc.vmap.vlane_stride

            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            tile_shape = local_tile_desc.get_mlir_shape(mlir_dtype)
            tile_stride = local_tile_desc.get_tile_stride()

            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, local_tile_desc, index)
            with self.override_buffer_cse(buffer=self.reductions_suffix):
                if self.welford_reduce_out is not None:
                    # Calc var and mean
                    sum, sqr_sum, _ = self.welford_reduce_out
                    reduction_numel = reduce(mul, self.ranges[self.reduction_depth:], 1)
                    divider = self.get_const_cse(float(reduction_numel), "f32")
                    mean = ops.truediv(sum, divider)
                    sqr_mean = ops.truediv(sqr_sum, divider)
                    mean_sqr = ops.mul(mean, mean)
                    variance = ops.sub(sqr_mean, mean_sqr)
                    m2 = ops.mul(variance, divider)
                    if self.current_node.node.origin_node: # FIXME: This is a temporary solution
                        value = mean
                    else:
                        value = m2
                # Store value to scratch pad
                ops._store(value, sram_var, sram_index_var, tile_shape, buffer_name=name)

            # Generate DMA instruction
            code = self.emit_transfer("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                      dram_shape, tile_shape, dram_stride, tile_stride, 0)
            self.reductions_suffix.writeline(common.DeferredLine(name, code))

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        self.indirect_symbols.add(str(index_var))  # record the bound indirect symbol
        return str(index_var)

    def _has_indirect(self, expr):
        return any(s.name in self.indirect_symbols for s in expr.free_symbols)

    def _index_expr(self, tile_desc, renamed_expression, index, base_vector_index):
        tile_size_per_lane = tile_desc.get_tile_size_per_lane()
        compute_vec_size = tile_desc.get_compute_vec_size()
        strides = tile_desc.get_tile_stride_per_lane()

        # Create vector index
        compute_vec = ops.broadcast(self.compute_idx, compute_vec_size)
        vector_index = ops.add(base_vector_index, compute_vec)

        # Create tile_dim index
        dim_list = []
        for idx in range(len(tile_size_per_lane)):
            # Prepare initial values
            offset = tile_desc.vmap.vlane_stride #* strides[idx]
            outer_sz = tile_desc.get_tile_size()[idx] // tile_desc.vmap.vlane_stride
            with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                div_coeff = self.get_const_cse(strides[idx], "index")
                mod_coeff = self.get_const_cse(tile_size_per_lane[idx], "index")
                vlane_stride_coeff = self.get_const_cse(tile_desc.vmap.vlane_stride, "index")
                vlane_outer_coeff = self.get_const_cse(outer_sz, "index")
                nr_vector_lane = self.get_const_cse(self.vector_lane, "index")
                vlane_coeff = self.get_const_cse(0, "i64")

                div_vec = ops.broadcast(div_coeff, compute_vec_size)
                mod_vec = ops.broadcast(mod_coeff, compute_vec_size)
                nr_vector_lane_vec = ops.broadcast(nr_vector_lane, compute_vec_size)
                vlane_stride_vec = ops.broadcast(vlane_stride_coeff, compute_vec_size)
                vlane_outer_vec = ops.broadcast(vlane_outer_coeff, compute_vec_size)

                # Prepare vlane offset (vidx)
                vlane_vec_size = 4
                vlane_vec = ops.broadcast(vlane_coeff, vlane_vec_size)

            dim = ops.remainder(ops.truncdiv(vector_index, div_vec), mod_vec)
            if idx == tile_desc.vmap.vlane_split_axis: # Need to add vector lane offset
                stride_dim = ops.remainder(dim, vlane_stride_vec)
                outer_dim = ops.remainder(ops.truncdiv(dim, vlane_stride_vec), vlane_outer_vec)
                dim = ops.add(stride_dim, ops.mul(outer_dim, nr_vector_lane_vec))

                with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                    vlane_offset = ops.vlane_offset(vlane_vec, vlane_vec, attributes={"vlane_offset": offset}, comment="vlane offset")
                    if compute_vec_size < self.var_info[vlane_offset][0]:
                        vlane_offset = ops.extract_strided_slice(vlane_offset, compute_vec_size)
                    vlane_offset = ops.index_cast(vlane_offset, "index")
                dim = ops.add(dim, vlane_offset)
            dim_list.append(dim)

        indices = [str(i) for i in index.free_symbols]
        for idx in indices:
            i = int(idx[5:])
            idx = self.itervar_cses[idx]
            index_vec = ops.broadcast(idx, compute_vec_size)
            offset = ops.add(index_vec, dim_list[i])
            dim_list[i] = offset

        arg_lists = []
        for arg in renamed_expression.args:
            if isinstance(arg, sympy.Integer):
                with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                    offset = self.get_const_cse(int(arg), "index")
                    offset_vec = ops.broadcast(offset, compute_vec_size)
                arg_lists.append(offset_vec)
            elif isinstance(arg, sympy.Mul):
                if isinstance(arg.args[0], sympy.Integer) and isinstance(arg.args[1], sympy.Symbol):
                    with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                        coeff = self.get_const_cse(int(arg.args[0]), "index")
                        coeff_vec = ops.broadcast(coeff, compute_vec_size)
                    result = ops.mul(dim_list[int(str(arg.args[1])[1:])], coeff_vec)
                    arg_lists.append(result)
                elif isinstance(arg.args[1], sympy.Integer) and isinstance(arg.args[0], sympy.Symbol):
                    with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
                        coeff = self.get_const_cse(int(arg.args[1]), "index")
                        coeff_vec = ops.broadcast(coeff, compute_vec_size)
                    result = ops.mul(dim_list[int(str(arg.args[0])[1:])], coeff_vec)
                    arg_lists.append(result)
                else:
                    raise NotImplementedError("Not supporting format")
            elif isinstance(arg, sympy.Symbol):
                arg_lists.append(dim_list[int(str(arg)[1:])])
            else:
                raise NotImplementedError("Not supporting format")
        if isinstance(renamed_expression, sympy.Symbol):
            arg_lists.append(dim_list[int(str(renamed_expression)[1:])])
        accum = arg_lists[0]
        for arg in arg_lists[1:]:
            accum = ops.add(accum, arg)
        return accum

    def index_expr(self, index, dtype):
        base_tile_desc = self.kernel_group.tile_desc
        if len(self.ranges) != self.reduction_depth:
            # FIXME. This is a temporary solution to get tile stride of the reduction case
            tile_desc = mlir_common.MLIRMultiDimTile(
                base_tile_desc.get_tile_size(),
                base_tile_desc.vmap.vector_lane,
                base_tile_desc.vmap.vlane_split_axis,
                base_tile_desc.vmap.vlane_stride,
                base_tile_desc.get_compute_vec_size(),
            )
            axis_order = list(range(len(tile_desc.get_tile_size())))
            axis_order = axis_order[1:] + axis_order[:1]  # Move the first axis to the end
            tile_desc.set_tile_size(tile_desc.get_tile_size(), axis_order)
        else:
            tile_desc = base_tile_desc
        compute_vec_size = tile_desc.get_compute_vec_size()

        tile_shape = f"memref<{compute_vec_size*self.vector_lane}xindex, 1>"
        vshape = f"vector<{compute_vec_size}xindex>"

        # Create base_vector index var
        c_type = "uint64_t"
        new_name = f"index_expr_{compute_vec_size}"
        if new_name not in self.global_vars_dict:
            self.header.writeline(f"{c_type} {new_name}_spad[{compute_vec_size*self.vector_lane}] __attribute__ ((section(\".spad\")));")
            self.gem5_header.writeline(f"{c_type} {new_name}_spad[{compute_vec_size}] __attribute__((aligned(64)));")
            self.global_vars.writeline(f"memref.global @{new_name}_spad : {tile_shape}")
            self.global_vars_dict[new_name] = dict()
        sram_var = self.spad_cse.generate(self.spad_buffer, f"memref.get_global @{new_name}_spad : {tile_shape}")

        # Initialize base vector
        if not self.base_vector_initialized:
            init_iter = self.register_var_cse("init_iter", 1, "index")
            parallel_map = f"affine.parallel (%{init_iter}) = ({0}) to ({compute_vec_size}) {{ // Base vector initializer"
            self.spad_buffer.writeline(parallel_map)
            with self.spad_buffer.indent():
                with self.override_buffer_cse(buffer=self.spad_buffer, cse=self.init_vec_cse):
                    init_vec = ops.broadcast(init_iter, 2)
                    ops._store(init_vec, sram_var, f"%{init_iter}", tile_shape)
            self.spad_buffer.writeline("}")
            self.base_vector_initialized = True
        base_vector_index = ops._load(compute_vec_size, "index", sram_var, "0", tile_shape)

        renamed_symbols = {symbol: "d"+str(symbol)[5:] for symbol in index.free_symbols}
        renamed_expression = index.subs(renamed_symbols)
        result = self._index_expr(tile_desc, renamed_expression, index, base_vector_index)
        return result

    def codegen_global_init(self):
        return self.global_vars

    def _bind_step(self, step):
        # Make `step` the current emit sink: route the body buffers to its buffers
        self.current_step = step
        self.applys = step.applys
        self.dma_loads = step.dma_loads
        self.dma_stores = step.dma_stores
        self.loads = step.loads
        self.compute = step.compute
        self.stores = step.stores

    def push_step(self):
        # New load->compute->store step; later emits land here, steps bridge via spad
        step = Step(
            applys=IndentedBuffer(),
            dma_loads=IndentedBuffer(), dma_stores=IndentedBuffer(),
            loads=IndentedBuffer(), compute=IndentedBuffer(), stores=IndentedBuffer(),
        )
        self.steps.append(step)
        self._bind_step(step)
        self.cse = self.cse.clone()  # share name counter, fresh dedup cache (region-safe)
        self.target_buffer_override.set(self.compute)
        self.target_cse_override.set(self.cse)
        return step

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
                    for reduction_loop in reductions.loops:
                        reduction_lines = reduction_loop.lines()
                        epilogue = reduction_loop.epilogue_line()
                        code.writelines(reduction_lines)
                        stack.enter_context(code.indent(attribute="{accumulation_loop=true}", suffix=epilogue))
                for step in self.steps:
                    code.splice(step.applys)
                    code.splice(step.dma_loads)
                    # Compute body -- only steps that have one get the loop + epilogue
                    if any(b.getvalue() for b in (step.loads, step.compute, step.stores)):
                        code.writelines(self.compute_body_loop.lines())
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(code.indent(attribute="{inner_loop=false}",suffix=self.compute_body_loop.epilogue_line()))
                            code.splice(step.loads)
                            code.splice(step.compute)
                            code.splice(step.stores)
                    code.splice(step.dma_stores)
            code.splice(self.reductions_suffix)
            # Non-outerloop end
        code.writeline(f"return")
        return code

    def make_choices(self, nodes, kernel_name):
        choices = []
        initial_tile_size = self.kernel_group.tile_desc.get_tile_size()
        prev_ranges = self.ranges
        prev_tail_threshold = self.kernel_group.tile_desc.tail_ratio_threshold

        # Allow more tail ratio during autotuning
        self.kernel_group.tile_desc.tail_ratio_threshold = 0.3

        if prev_ranges == [1] or len(prev_ranges) == 0:
            return choices
        #if len(initial_tile_size) < 2:
        #    return choices # Can't autotune for 1-D tile size

        for vlane_stride in [2, 4, 8]:
            self.kernel_group.tile_desc.set_tile_size(initial_tile_size)
            self.kernel_group.tile_desc.vmap.vlane_stride = vlane_stride
            prevent_infinite_loop = 0

            # Get the dimension to increase
            candidate_axes = [
                axis for axis, constr in enumerate(self.kernel_group.tile_desc.tile_constraint)
                if not constr.fixed
            ]
            search_space = set()

            # Try initial tile size
            self.reset(None)
            try:
                src_code, meta_code = super().codegen_nodes(nodes, kernel_name)
            except mlir_common.RecompileSignal:
                continue
            current_tile_sz = tuple(self.kernel_group.tile_desc.get_tile_size())
            search_space.add(current_tile_sz)

            logger.debug(f"Auto-tune: Trying tile size: {list(current_tile_sz)}, vlane_stride: {self.kernel_group.tile_desc.vmap.vlane_stride}, split_axis: {self.kernel_group.tile_desc.vmap.vlane_split_axis}")
            self._prepare_simulator_headers(src_code)
            bench_runner = self.run_bench(nodes, kernel_name, src_code)
            choices.append((bench_runner, src_code, meta_code, current_tile_sz, self.kernel_group.tile_desc.vmap.vlane_stride))

            while prevent_infinite_loop < 10 and candidate_axes:
                for axis in list(candidate_axes):
                    prev_tile_sz = self.kernel_group.tile_desc.get_tile_size()

                    # If tile size is maximized for this axis, remove from candidate axes
                    if prev_tile_sz[axis] >= prev_ranges[axis] * 2 or prev_tile_sz[axis] >= 2 ** 13:
                        candidate_axes.remove(axis)
                        self.reset(None)
                        continue

                    # Try increase tile size for this axis
                    try:
                        self.kernel_group.tile_desc.scale_tile_dim(axis, prev_ranges[axis], 2)
                        self.reset(None)
                        src_code, meta_code = super().codegen_nodes(nodes, kernel_name)
                    except (extension_codecache.TileSizeError, mlir_common.RecompileSignal):
                        candidate_axes.remove(axis)
                        self.reset(None)
                        continue
                    current_tile_sz = tuple(self.kernel_group.tile_desc.get_tile_size())

                    # FIXME. How to intergrate this constraint to tile system?
                    pad = self.kernel_group.tile_desc.vmap.get_used_vlane(current_tile_sz) * self.kernel_group.tile_desc.vmap.vlane_stride
                    vlane_size = current_tile_sz[self.kernel_group.tile_desc.vmap.vlane_split_axis]
                    if vlane_size > pad and vlane_size % pad:
                        prevent_infinite_loop += 1
                        continue

                    # If tile size is converged for this axis, remove from candidate axes
                    if current_tile_sz in search_space:
                        candidate_axes.remove(axis)
                        continue

                    # Add this choice
                    search_space.add(current_tile_sz)
                    logger.debug(f"Auto-tune: Trying tile size: {list(current_tile_sz)}, vlane_stride: {self.kernel_group.tile_desc.vmap.vlane_stride}, split_axis: {self.kernel_group.tile_desc.vmap.vlane_split_axis}")
                    self._prepare_simulator_headers(src_code)
                    bench_runner = self.run_bench(nodes, kernel_name, src_code)
                    choices.append((bench_runner, src_code, meta_code, self.kernel_group.tile_desc.get_tile_size(), self.kernel_group.tile_desc.vmap.vlane_stride))
                    prevent_infinite_loop += 1
        self.kernel_group.tile_desc.prev_tail_threshold = prev_tail_threshold
        return choices

    def autotune(self, *args):
        def get_cycle(choice, subprocess_timeout_sec=None):
            bench_runner = choice[0]
            for n_try in range(extension_config.codegen_autotune_max_retry): # TODO: make simple
                try:
                    if subprocess_timeout_sec is not None:
                        out = bench_runner(
                            autotune_subprocess_timeout_sec=subprocess_timeout_sec
                        )
                    else:
                        out = bench_runner()
                    return out[-1]
                except (extension_codecache.SpadOverflowError, RuntimeError):
                    return float("inf")
            return float("inf") # Exceeded maximum number of autotuning attempts
        choices = self.make_choices(*args)
        if len(choices) == 0: # Can't autotune
            return [None, None, None]

        slack_sec = float(extension_config.codegen_autotune_wall_slack_sec)

        # Get cycle time for each choice
        # Show progress bar only when CONFIG_DEBUG_MODE is off
        show_progress = not extension_config.CONFIG_DEBUG_MODE
        with ProgressBar("[Auto-tune] Running benchmarks", silent_mode=not show_progress) if show_progress else contextlib.nullcontext():
            results = [float("inf")] * len(choices)
            baseline_wall = None
            parallel_from = 0

            for idx, choice in enumerate(choices):
                t0 = time.perf_counter()
                c = get_cycle(choice, None)
                elapsed = time.perf_counter() - t0
                results[idx] = c
                parallel_from = idx + 1
                if c != float("inf"):
                    baseline_wall = elapsed
                    break

            pending = choices[parallel_from:]
            if baseline_wall is not None and pending:
                timeout_sec = baseline_wall + slack_sec
                workers = min(8, len(pending), os.cpu_count())
                executor = ThreadPoolExecutor(max_workers=workers)
                try:
                    tail = list(
                        executor.map(
                            lambda ch: get_cycle(ch, timeout_sec), pending
                        )
                    )
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
                results[parallel_from : parallel_from + len(tail)] = tail

        min_idx = results.index(min(results))
        if min(results) == float("inf"):
            raise RuntimeError("Failed to find optimal tile size...")

        self._log_autotune_result(choices[min_idx], results[min_idx])

        optimal_src_code, meta_code, loop_size = choices[min_idx][1], choices[min_idx][2], choices[min_idx][-1]
        return optimal_src_code, meta_code, loop_size

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
                "arg_attributes" : arg_attributes,
                "autotune" : True,
                "loop_size" : self.loop_size,
                "origins" : {str(i) for node in nodes for i in node.node.origins},
            },
            source_code=src_code,
        )
        dummy_inputs = [rand_strided(meta.sizes,meta.strides,dtype=meta.dtype, extra_size=meta.offset).to(device=nodes[0].get_device()) for meta in bmreq.input_tensor_meta]
        dummy_outputs = [rand_strided(meta.sizes,meta.strides,dtype=meta.dtype, extra_size=meta.offset).to(device=nodes[0].get_device()) for meta in bmreq.output_tensor_meta]
        return bmreq.make_run_fn(dummy_inputs, dummy_outputs)

    def _log_autotune_result(self, best_choice, best_cycle):
        logger.debug(
            f"Auto-tune: Optimal tile size: {list(best_choice[3])}, "
            f"vlane_stride: {best_choice[4]}, "
            f"cycles: {best_cycle}"
        )

    def codegen_nodes(self, nodes, kernel_name):
        src_code, meta_code = super().codegen_nodes(nodes, kernel_name)
        self._prepare_simulator_headers(src_code)
        if "autotune" in extension_config.codegen_mapping_strategy and extension_config.pytorchsim_timing_mode:
            # Use temporaries: autotune returns [None, None, None] when it cannot
            # autotune (e.g. a size-1 pointwise kernel with ranges == [1]), and
            # unpacking into meta_code would clobber the valid arg_attributes that
            # the fall-through below returns.
            optimal_src_code, optimal_meta_code = self.autotune(nodes, kernel_name)[:2]
            if optimal_src_code is not None:
                return optimal_src_code, optimal_meta_code
        return src_code, meta_code

    def _prepare_simulator_headers(self, src_code):
        spad_end_symbol = "int spad_end[0] __attribute__ ((section(\".spad\")));\n"
        spad_section_end_symbol = (
            f"int spad_section_end[0] __attribute__ ((section(\".spad\"), aligned({self.spad_info['spad_size']*self.vector_lane})));"
        )
        spike_content = self.header.getvalue() + spad_end_symbol + spad_section_end_symbol
        gem5_content = self.gem5_header.getvalue()
        extension_codecache.store_header(src_code, spike_content, gem5_content)

    def get_arg_info(self, name):
        arg_info = dict()
        arg_info.update(V.graph.graph_inputs)
        arg_info.update({i.get_name(): i for i in V.graph.buffers})
        return arg_info[name]

    def get_dma_info(self, name, index, broadcast=True, store_reduction=False, buffer=None): # Need more argument?
        """
        A tile descriptor exists that is configured on a kernel group
        DMA desc should be adjusted according to buffer.
        Therefore, this function shoulde determin DRAM, SRAM stride and
        vectorlane mapping policy
        """
        # Use loads as default
        if buffer is None:
            buffer = self.applys if not self._has_indirect(index) else self.dma_loads

        # TODO.
        kg_tile_desc = self.kernel_group.tile_desc
        # Note: index could contain symbols that represent dynamic axies
        # Extract dimension of index(e.g, index0, index1)
        local_dims = [int(str(i)[5:]) for i in index.free_symbols if "index" in str(i)]
        total_dims =  [int(str(i)[5:]) for i in self.itervars]
        local_tile_desc = mlir_common.MLIRMultiDimTile([1], self.vector_lane)
        local_dims.sort() # Assume that smaller index is placed in the outer loop
        indirect_syms = [s for s in index.free_symbols if s.name in self.indirect_symbols]
        index = index.subs({s: 0 for s in indirect_syms}, simultaneous=True)
        indirect_dims = [f"{i}" for i in indirect_syms]

        # axis-split + graph-copy linearize aligned floor/mod upstream. Anything that
        # reaches here still carrying floor/mod (store-side ModularIndexing,
        # reduction-axis floor/mod, incompatible-radix views) would be silently
        # mis-strided in the dram_stride computation below, so fail loudly instead.
        if index.has(FloorDiv) or index.has(ModularIndexing):
            raise NotImplementedError(
                f"Unlinearized floor/mod in DMA index: {index}. axis-split/graph-copy "
                f"did not eliminate it; this view is unsupported "
                f"(see docs/axis-split-scheduling.md)."
            )

        # Reduction can have two type of tile size
        if broadcast and (total_dims != local_dims or (self.reduction_depth!=len(total_dims) and total_dims[:self.reduction_depth] == local_dims)):
            local_dims = total_dims # Brodatcast tile shape

        with self.override_buffer_cse(buffer=buffer, cse=self.apply_cse):
            index_var = self.parse_indices(index, indirect_dims=indirect_dims, comments=f"// store_reduction={store_reduction}")

        if kg_tile_desc.vmap.vlane_split_axis in local_dims:
            local_vlane_split_axis = local_dims.index(kg_tile_desc.vmap.vlane_split_axis)
        else:
            local_vlane_split_axis = max(len(local_dims) - 1, 0)

        # Case 0. Tile is 0-D scalar
        if len(local_dims) == 0:
            if not store_reduction:
                local_tile_desc.set_tile_size([kg_tile_desc.get_used_vlane() * kg_tile_desc.vmap.vlane_stride])         # Force it to use vector instruction.
                local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis    # last axis
                local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
            else:
                local_tile_desc.set_tile_size([1])
                local_tile_desc.vmap.vlane_split_axis = 0
                local_tile_desc.vmap.vlane_stride = 1
            dram_stride = [0] # Edge case
        # Case 1. Tile is 1-D vector type
        elif len(local_dims) == 1 and len(local_dims) <= self.reduction_depth:
            local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(local_dims[0])])
            local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
            local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
        # Case 2. Tile is 1-D vector type with reduction
        elif len(local_dims) == 1 and len(local_dims) == self.reduction_depth + 1:
            local_tile_desc.set_tile_size([1, kg_tile_desc.get_dim_size(local_dims[0])])
            local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis + 1
            local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
        # Case 3. Tile is 2-D tile
        elif len(local_dims) == 2:
            is_reduction = self.reduction_depth == 1 and not store_reduction
            if is_reduction:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims], [1, 0])
                local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
            else:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
                local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
        # Case 3. Tile is 3-D tile
        elif len(local_dims) == 3:
            is_reduction = self.reduction_depth < 3 and not store_reduction
            if is_reduction:
                axis_order = [1, 2, 0] if self.get_nr_rdim()==1 else [2, 1, 0]
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims], axis_order)
                local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
            else:
                local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
                local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
                local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
        # Case 4. Tile is 4-D tile (e.g., Convolution epilogue)
        elif len(local_dims) == 4:
            is_reduction = self.reduction_depth < 3 and not store_reduction
            if is_reduction:
                raise NotImplementedError("Currently not implemented... ;)")
            local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
            local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
            local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride
        else:
            local_tile_desc.set_tile_size([kg_tile_desc.get_dim_size(dim) for dim in local_dims])
            local_tile_desc.vmap.vlane_split_axis = local_vlane_split_axis
            local_tile_desc.vmap.vlane_stride = kg_tile_desc.vmap.vlane_stride

        # Calculate dram stride in local tile-dim order.
        # This keeps dram/sram stride rank aligned with tile rank.
        local_dim_to_axis = {dim: axis for axis, dim in enumerate(local_dims)}
        dram_stride = [0] * local_tile_desc.get_nr_dim()
        if index.is_Symbol:
            dim_idx = int(str(index)[5:])
            if dim_idx in local_dim_to_axis:
                dram_stride[local_dim_to_axis[dim_idx]] = 1
        elif index.is_Number:
            pass
        else:

            dram_dict = defaultdict(list)
            for arg in index.as_ordered_terms():
                coeff, dim = arg.as_coeff_mul()
                if len(dim) == 0:
                    continue
                real_dim = list(dim[0].free_symbols)[0]
                dram_dict[str(real_dim)].append(coeff)

            # Add missing dims if not added
            max_dim = len(self.ranges) if not store_reduction else len(self.ranges) - 1
            for i in range(max_dim):
                target_dim = f"index{i}"
                if sympy.Symbol(target_dim) not in index.free_symbols:
                    dram_dict[target_dim] = [0]
            sorted_keys = sorted(dram_dict.keys())
            dram_stride = sum((dram_dict[key] for key in sorted_keys), [])

        # FIXME. It will be nice to modify node instead of this exception handling...
        if len(self.itervars) == 1 and self.reduction_depth == 0:
            # In case of reduction loop only case, we will add dummy loop so shift it once
            dram_stride = [0] + dram_stride[:-1]

        # Return the tile-axis -> loop-dim map (local_dims) so load()/store() can pass it to
        # _masked_bounds for the per-dim [low, high) clamp (it needs each tile axis' loop iv).
        return local_tile_desc, index_var, dram_stride, local_dims

    _FILL_BITVIEW = {torch.float32: torch.int32, torch.float16: torch.int16,
                     torch.bfloat16: torch.int16, torch.float64: torch.int64}

    def _masked_fill_bits(self, dtype, index):
        """Raw bits for the masked-DMA tail fill = the consuming reduction's identity
        (reduction_init; sum->0, max->-inf, ...) in the LOAD dtype, 0 for a non-reduction
        load. Log-sum-exp exception: a sum reducing exp(input) fills its PRIMARY input's
        tail with -inf (exp(-inf)=0); broadcast operands keep their finite identity."""
        node = getattr(self, "current_node", None)
        if node is None or getattr(node, "node", None) is None or not node.node.get_reduction_type():
            return 0
        rtype = node.node.get_reduction_type()
        is_primary = bool(set(self.itervars[self.reduction_depth:]) & index.free_symbols)
        if rtype == "sum" and is_primary and any(self._origin_is_exp(o) for o in node.node.origins):
            init = "-inf"
        else:
            init = reduction_init(rtype, dtype)
        val = {"-inf": float("-inf"), "inf": float("inf")}.get(init, init)
        if isinstance(val, str):     # e.g. welford_reduce -> "0.0"
            val = float(val)
        t = torch.tensor(val, dtype=dtype)
        view = self._FILL_BITVIEW.get(dtype)
        bits = int(t.view(view).item()) if view is not None else int(t.item())
        return bits & ((1 << (t.element_size() * 8)) - 1)

    def _masked_bounds(self, name, index, dram_stride, local_tile_desc, is_load, buffer, local_dims):
        """Per tile-axis [low, high) clamp for a masked DMA. Per axis the valid GLOBAL range
        is [glo, ghi) (store: [0, output_extent); padded load: [pad, pad + input_extent)), and
        _emit_clamp turns it into the tile-local low/high SSA vars. Returns [(tile_axis, low_var, high_var), ...].
        """
        tile_size = local_tile_desc.get_tile_size()
        const = int(index.as_coeff_Add()[0]) if not index.is_number else int(index)
        # index const = -sum(pad_d * dram_stride[d]); recover per-dim pad greedily (desc stride).
        pad = [0] * len(tile_size)
        if is_load and const < 0:
            rem = -const
            for d in sorted(range(len(dram_stride)), key=lambda x: -dram_stride[x]):
                s = dram_stride[d]
                if s > 0 and d < len(pad):
                    pad[d] = rem // s
                    rem -= pad[d] * s
        in_shape, in_stride = self.buffer_types[name][2], self.buffer_types[name][3]
        axes = []
        for d, k in enumerate(local_dims):
            if d >= len(tile_size) or k >= len(self.ranges):
                continue
            iv = str(self.itervars[k])
            # A padded load (const < 0, i.e. index shifted by -pad) reads a smaller input:
            # clamp per-dim to [pad_d, pad_d + input_extent_d). Every other DMA (stores and
            # non-padded/contiguous loads, including collapsed tensors) is valid over the
            # whole loop extent -> only the trailing tail needs clamping.
            if is_load and const < 0:
                ext = next((int(in_shape[j]) for j, st in enumerate(in_stride)
                            if int(st) == int(dram_stride[d])), None)
                glo, ghi = (pad[d], pad[d] + ext) if ext is not None else (0, int(self.ranges[k]))
            else:
                glo, ghi = 0, int(self.ranges[k])
            axes.append((d, iv, glo, ghi, int(tile_size[d])))
        return self._emit_clamp(axes, buffer)

    def _emit_clamp(self, axes, buffer):
        """Emit the per-axis dynamic clamp. axes: [(tile_axis, base_iv, glo, ghi, tile), ...]
        -- low = max(0, glo - base), high = min(tile, ghi - base) as affine.max/affine.min of
        the loop iv so the last partial tile and the pad borders fall out per iteration.
        Returns [(tile_axis, low_var, high_var), ...] for the non-trivial axes only."""
        result = []
        for d, iv, glo, ghi, tile in axes:
            if glo == 0 and ghi % tile == 0:      # every tile fully valid -> no clamp
                continue
            high_var = self.apply_cse.generate(
                buffer, f"affine.min affine_map<(d0) -> ({tile}, {ghi} - d0)>(%{iv})")
            self.register_var_info(high_var, [1, "index"])
            if glo > 0:
                low_var = self.apply_cse.generate(
                    buffer, f"affine.max affine_map<(d0) -> (0, {glo} - d0)>(%{iv})")
                self.register_var_info(low_var, [1, "index"])
            else:
                low_var = self.get_const_cse(0)
            result.append((d, low_var, high_var))
        return result

    def emit_transfer(self, dma_type_name, vlane_split_axis, vlane_stride, mlir_dtype,
                      dram_var, dram_index_var, sram_var, sram_index_var,
                      dram_shape, tile_shape, dram_stride, tile_stride, padding,
                      subtile_size=None, async_type=None, offset=None, masked_bounds=None, masked_fill=0,
                      accumulate=False, acc_float=False):
        """Emit a generic togsim.transfer op for a DMA whose access exceeds the
        4D Gemmini descriptor limit. Carries the full N-D access (dram/tile
        strides + shapes) plus the SSA operands a memref.dma_start needs
        (dma_type / vlane_split_axis / vlane_stride), so the decompose pass
        (passes/decompose_transfer.py) is purely mechanical: it peels the excess
        dims into a loop of <=4D memref.dma_start, reusing these operands.

        Operand prep uses the read/write cache+counter for the dma_type enum and
        CSE'd vlane consts so the transfer is self-contained; togsim is an
        unregistered dialect -> generic form.
        """
        dma_key = (vlane_split_axis, vlane_stride, mlir_dtype)
        if dma_type_name == "MVIN" and dma_key in self.dma_read_cache:
            dma_type, vsa, vst = self.dma_read_cache[dma_key]
        elif dma_type_name == "MVOUT" and dma_key in self.dma_write_cache:
            dma_type, vsa, vst = self.dma_write_cache[dma_key]
        else:
            vsa = self.get_const_cse(vlane_split_axis)
            vst = self.get_const_cse(vlane_stride)
            if dma_type_name == "MVIN":
                dma_type = self.get_const_cse(DMA_TYPE[f"{dma_type_name}{self.dma_read_counter}"])
                self.dma_read_counter += 1
                self.dma_read_cache[dma_key] = [dma_type, vsa, vst]
            else:
                dma_type = self.get_const_cse(DMA_TYPE[f"{dma_type_name}{self.dma_write_counter}"])
                self.dma_write_cache[dma_key] = [dma_type, vsa, vst]
        tag = self.get_tag_cse()
        zero_cse = self.get_const_cse(0)
        # vlane_split_axis is carried as a VALUE attr (not an SSA operand) because the
        # decompose pass must remap it: collapsing unit tile dims renumbers the axes,
        # so the descriptor's vlane axis index changes and the pass rebuilds the const.
        attrs = (
            f'dma_kind = "{dma_type_name}", '
            f'vlane_split_axis = {int(vlane_split_axis)} : i64, '
            f'dram_stride = {dram_stride}, tile_stride = {tile_stride}, '
            f'padding = {int(padding)} : i64'
        )
        if subtile_size:
            av = int(async_type) if async_type is not None else 1
            attrs += f', subtile_size = {list(subtile_size)}, async = {av} : i64'
        if accumulate:   # index_add: MVOUT does out[idx] += val (float or integer add)
            attrs += f', accumulate = true'
            if acc_float:
                attrs += f', acc_float = true'
        # operands: dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vlane_stride [, offset spad]
        operands = (f'%{dram_var}, %{dram_index_var}, %{sram_var}, %{zero_cse}, '
                    f'%{tag}, %{zero_cse}, %{dma_type}, %{vst}')
        optypes = f'{dram_shape}, index, {tile_shape}, index, memref<1xi32>, index, index, index'
        if offset is not None:  # indirect: per-position offset spad (decompose lifts it to a symbol attr)
            offset_buf, offset_type, offset_stride = offset
            operands += f', %{offset_buf}'
            optypes += f', {offset_type}'
            attrs += f', indirect = true, offset_stride = {int(offset_stride)} : i64'
        # masked-DMA dynamic clamp: append (low, high) index operands per clamped tile axis;
        # masked_axes names the tile axis of each pair so the lowering writes the runtime
        # values into the descriptor's dim_low/dim_high before the DMA. See _masked_bounds.
        if masked_bounds:
            axes = [d for d, _lo, _hi in masked_bounds]
            for _d, lo, hi in masked_bounds:
                operands += f', %{lo}, %{hi}'
                optypes += ', index, index'
            attrs += f', masked_axes = {axes}'
            # box-excluded positions are filled with the consuming reduction's identity
            # (0/1/-inf/+inf, per dtype); 0 for non-reduction loads. See _masked_fill_bits.
            attrs += f', masked_fill = {int(masked_fill)} : i64'
        return f'"togsim.transfer"({operands}) {{{attrs}}} : ({optypes}) -> ()'

    def allocate_sram_buffer(self, dtype, dram_name, tile_desc, raw_index, buffer=None, forced_name=None):
        c_type = mlir_common.DTYPE_TO_C[dtype]
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        tile_numel_per_lane = tile_desc.get_numel_per_lane()
        tile_shape = tile_desc.get_mlir_shape(mlir_dtype)
        # Make sure each lane's buffer has at least two element
        tile_size = max(tile_numel_per_lane, 2) * self.vector_lane

        if buffer is None:
            buffer = self.spad_buffer

        if dram_name not in self.global_vars_dict:
            self.global_vars_dict[dram_name] = dict()

        if str(raw_index) not in self.global_vars_dict[dram_name]:
            new_name = f"buf{self.spadbuf_counter}_spad" if forced_name is None else f"{forced_name}_spad"
            self.spadbuf_counter+=1
            # Add definition to header
            self.header.writeline(f"{c_type} {new_name}[{tile_size // self.vector_lane}] __attribute__ ((section(\".spad\")));")
            self.gem5_header.writeline(f"{c_type} {new_name}[{tile_size}] __attribute__((aligned(64)));")
            self.global_vars.writeline(f"memref.global @{new_name} : {tile_shape}")
            self.global_vars_dict[dram_name][str(raw_index)] = new_name
        else:
            new_name = self.global_vars_dict[dram_name][str(raw_index)]
        return new_name

    def get_scratchpad_buffer(self, dtype, dram_name, tile_desc, raw_index, buffer=None):
        if buffer is None:
            buffer = self.spad_buffer

        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        tile_shape = tile_desc.get_mlir_shape(mlir_dtype)
        new_name = self.allocate_sram_buffer(dtype, dram_name, tile_desc, raw_index, buffer=buffer)
        sram_var = self.spad_cse.generate(buffer, f"memref.get_global @{new_name} : {tile_shape}")

        zero_cse = self.get_const_cse(0)
        sram_index_var = ",".join([f"%{zero_cse}"] * tile_desc.get_nr_dim())
        return sram_var, sram_index_var

    def get_const_cse(self, value, dtype="index") -> common.CSEVariable:
        # Why not use ops.constant? Because there are some cases that can't use ops (e.g., def_dma_op)
        # Type convert
        if value in ["inf", "-inf", "nan"]:
            value = f"0x{mlir_common.MLIR_INF[value][dtype]:x}"
        elif dtype[0] == "f":
            value = float(value)
        else:
            value = int(value)
        key = str(value)+dtype
        if key not in self.consts:
            self.consts[key] = self.const_cse.generate(self.const_buffer, f"arith.constant {value} : {dtype}")
            self.register_var_info(self.consts[key], [1, dtype])
        return self.consts[key]

    def get_tag_cse(self, value=None, shape="memref<1xi32>"):
        if value is None:
            value = self.dma_tag_id
            self.dma_tag_id += 1
        if value not in self.tags:
            self.tags[value] = self.alloc_cse.generate(self.alloc_buffer, f"memref.alloc() : {shape} // {value}")
        return self.tags[value]

    def get_mask(self):
        if self.compute_body_loop.size % self.compute_body_loop.step == 0:
            return None, None
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()
        mask_shape = f"vector<{compute_vec_size}xi1>"

        with self.override_buffer_cse(buffer=self.const_buffer, cse=self.const_cse):
            upper_bound = ops.constant(self.compute_body_loop.size, "index")
            step_vec = ops.step(self.compute_body_loop.step, "index")

        with self.override_buffer_cse(buffer=self.compute, cse=self.mask_cse):
            gap = ops.sub(upper_bound, self.compute_idx)
            gap_vec = ops.broadcast(gap, self.compute_body_loop.step)
            mask_var = ops.lt(step_vec, gap_vec)
        return mask_shape, mask_var

    def convert_indirect_indexing(self, index :sympy.Expr):
        if not self._has_indirect(index):
            return index, None

        # Process start
        indirect_dims = [str(dim) for dim in index.free_symbols if str(dim) in self.indirect_symbols]
        indirect_dims.sort()
        first_dim = indirect_dims[0]
        spad_vars = dict()
        compute_dependecy = any([target_dim not in self.spad_buffer_dict for target_dim in indirect_dims])
        # Store each newly-produced indirect index into spad, in its producing step
        for target_dim in indirect_dims:
            if target_dim in self.spad_buffer_dict:
                continue
            var_info = [v for k, v in self.var_info.items() if str(k) == target_dim][0]
            dtype = mlir_common.MLIR_TO_DTYPE[var_info[1]]
            local_tile_desc = self.kernel_group.tile_desc
            tile_numel_per_lane = local_tile_desc.get_numel_per_lane()
            tile_shape = local_tile_desc.get_mlir_shape(var_info[1])
            tile_vec = local_tile_desc.get_compute_vec_size()
            vshape = f"vector<{var_info[0]}x{var_info[1]}>"
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, target_dim, local_tile_desc, target_dim)
            self.spad_buffer_dict[target_dim] = [sram_var, local_tile_desc.get_tile_size(), tile_numel_per_lane, sram_index_var, tile_shape, vshape]
            target_var = self.cse.varname_map[target_dim]
            compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])
            with self.override_buffer_cse(buffer=self.stores):
                ops._store(target_var, sram_var, compute_index_var, tile_shape)

        # Offset build runs after the index is in spad -> own step when just produced
        if compute_dependecy:
            self.push_step()

        # Single indirect dim: the raw index IS the offset; the MVIN applies offset_stride (CONFIG4)
        if len(indirect_dims) == 1:
            offset_stride = 1
            for arg in list(index.args):
                if not self._has_indirect(arg):
                    continue
                if arg.is_Mul and arg.args[0].is_number:
                    offset_stride = int(arg.args[0])
                index = index.replace(arg, 0)
            sram_var, _, _, _, tile_shape, _ = self.spad_buffer_dict[first_dim]
            return index, (sram_var, tile_shape, offset_stride)

        # Multi indirect dim: sum the strided indices in the compute loop (chunked by compute_vec_size)
        local_tile_desc = self.kernel_group.tile_desc
        compute_vec_size = local_tile_desc.get_compute_vec_size()
        for target_dim in indirect_dims:
            sram_var, _, _, sram_index_var, tile_shape, vshape = self.spad_buffer_dict[target_dim]
            mlir_dtype = vshape.split("x")[1][:-1]
            compute_index_var = ",".join(sram_index_var.split(",")[:-1] + [f"%{self.compute_idx}"])
            with self.override_buffer_cse(buffer=self.loads):
                spad_vars[target_dim] = ops._load(compute_vec_size, mlir_dtype, sram_var, compute_index_var, tile_shape)
        with self.override_buffer_cse(buffer=self.compute):
            for arg in index.args:
                if not self._has_indirect(arg):
                    continue
                if arg.is_Mul and arg.args[0].is_number:
                    coeff_dtype = self.var_info[spad_vars[str(arg.args[1])]][1]
                    coeff = self.get_const_cse(int(arg.args[0]), coeff_dtype)
                    spad_vars[str(arg.args[1])] = ops.mul(spad_vars[str(arg.args[1])], coeff)
                index = index.replace(arg, 0)
            for dim, var in spad_vars.items():
                if dim == first_dim:
                    continue
                spad_vars[first_dim] = ops.add(spad_vars[first_dim], var)
        # Summed offset goes to a DEDICATED spad (not an index buffer) to avoid clobbering a live index
        var_info = [v for k, v in self.var_info.items() if str(k) == first_dim][0]
        dtype = mlir_common.MLIR_TO_DTYPE[var_info[1]]
        off_shape = local_tile_desc.get_mlir_shape(var_info[1])
        off_sram, off_index = self.get_scratchpad_buffer(
            dtype, "indirect_offset_" + first_dim, local_tile_desc, "indirect_offset_" + first_dim)
        off_compute_index = ",".join(off_index.split(",")[:-1] + [f"%{self.compute_idx}"])
        with self.override_buffer_cse(buffer=self.stores):
            ops._store(spad_vars[first_dim], off_sram, off_compute_index, off_shape)
        self.push_step()  # offset-build compute loop must finish before the gather reads it
        return index, (off_sram, off_shape, 1)
