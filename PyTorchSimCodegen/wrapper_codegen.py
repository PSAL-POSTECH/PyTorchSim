"""The Python wrapper module Inductor generates around the compiled kernels.

Emits the header the wrapper needs (torchsim_compile,
the functional-verify calls) and walks the wrapper IR lines once.
"""
import contextlib

import torch
from torch._inductor.codegen import wrapper
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from typing import Optional

from Simulator import functional_verify as _func_verify

from . import codecache, kernel_spec


def _writes(kernel_name, position):
    """Does `kernel_name` write the tensor argument at `position`? True when unrecorded."""
    if kernel_name is None:
        return True
    return kernel_spec.writes_arg(kernel_name, position)


def _mutated(line):
    """Buffer names a non-kernel wrapper line (a fallback) writes in place, from Inductor's IR."""
    node = getattr(line, "node", None)
    if node is None:
        return ()
    try:
        return tuple(node.get_mutation_names())
    except Exception:  # noqa: BLE001
        return ()


class TritonNPUWrapperCodegen(wrapper.PythonWrapperCodegen):
    """The npu wrapper: kernels are compiled by torchsim_compile and called as plain functions."""

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

    def wrap_kernel_call(self, name, call_args):
        """The kernel call line, with sympy call args rendered as strings first."""
        return super().wrap_kernel_call(
            name, self.prepare_triton_kernel_call(call_args))

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
                from PyTorchSimFrontend.config import setup_logger
                from Simulator import functional_verify as _fverify
                from torch._inductor.select_algorithm import extern_kernels
                from {codecache.__name__} import torchsim_compile

                _logger = setup_logger("PyTorchSimCodegen.generated_wrapper")

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = AsyncCompile()
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
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

            if _func_verify.enabled():
                gm = getattr(V.graph, "module", None)
                if gm is not None:
                    gid = _func_verify.register_graph(gm)
                    in_names = list(V.graph.graph_inputs.keys())
                    self.prefix.writeline(
                        f"_fverify.verify_init({gid}, [{', '.join(in_names)}])")

            self.codegen_inputs()
            self.codegen_input_size_asserts()

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
        self.writeline(self.wrap_kernel_call(kernel_name, call_args))
        return

    def generate(self, is_inference):
        result = IndentedBuffer()

        self._fverify_seen = set()
        self._fverify_last = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if torch._inductor.config.allow_buffer_reuse:
                self.estimate_peak = wrapper.EfficientPeakEstimate()
            self.memory_plan_reuse()
            with self.set_writeline(self.wrapper_call.writeline):
                for line in self.lines:
                    if isinstance(line, wrapper.MemoryPlanningLine):
                        line.codegen(self.wrapper_call)
                    elif isinstance(line, wrapper.KernelCallLine):
                        self.wrapper_call.writeline(self.wrap_kernel_call(line.kernel_name, line.call_args))
                        if _func_verify.enabled():
                            self._fverify_emit_checks(line.call_args, id(line),
                                                      line.kernel_name)
                    else:
                        if isinstance(line, wrapper.WrapperLine):
                            line.codegen(self.wrapper_call)
                            if _func_verify.enabled():
                                self._fverify_emit_mutation_checks(line)
                        else:
                            self.wrapper_call.writeline(line)
            output_refs = self.get_output_refs()
            self.mark_output_type()
            self.generate_return(output_refs)

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

    def _fverify_last_writer(self):
        """{buffer name: id of the last wrapper line that writes it}.

        A buffer is checked after its last writer, since one op can span several kernels.
        """
        last = {}
        for line in self.lines:
            if isinstance(line, wrapper.KernelCallLine):
                for pos, a in enumerate(line.call_args):
                    if not (isinstance(a, str) and a.strip().isidentifier()):
                        continue
                    if not _writes(line.kernel_name, pos):
                        continue
                    last[a.strip()] = id(line)
                continue
            for name in _mutated(line):
                last[name] = id(line)
        return last

    def _fverify_emit_checks(self, call_args, line_id=None, kernel_name=None):
        """Emit a verify_check after this kernel for each buffer it writes and finishes.

        Only written arguments are checked: a reused buffer that is only read carries another origin.
        """
        if self._fverify_last is None:
            self._fverify_last = self._fverify_last_writer()
        for pos, a in enumerate(call_args):
            if not isinstance(a, str):
                continue
            name = a.strip()
            if not name.isidentifier():
                continue
            if not _writes(kernel_name, pos):
                continue
            self._fverify_check_one(name, line_id)

    def _fverify_emit_mutation_checks(self, line):
        """The same check, after a fallback that finishes a buffer in place."""
        if self._fverify_last is None:
            self._fverify_last = self._fverify_last_writer()
        for name in _mutated(line):
            self._fverify_check_one(name, id(line))

    def _fverify_check_one(self, name, line_id):
        """One `verify_check` for `name`, if this line is its last writer."""
        if name in self._fverify_seen:
            return
        if line_id is not None and self._fverify_last.get(name) != line_id:
            return
        self._fverify_seen.add(name)
        if name in V.graph.graph_inputs:
            return
        try:
            buf = V.graph.get_buffer(name)
        except Exception:
            buf = None
        if buf is None:
            return
        origin = getattr(buf, "origin_node", None)
        if origin is None:
            return
        op = str(getattr(origin, "target", "?"))
        self.wrapper_call.writeline(
            f'_fverify.verify_check({name}, "{name}", "{origin.name}", "{op}")')
