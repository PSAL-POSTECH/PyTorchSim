"""The Python wrapper module Inductor generates around the compiled kernels.

Emits the header the wrapper needs (torchsim_compile,
the functional-verify calls) and walks the wrapper IR lines once.
"""
import contextlib

import torch
from torch._inductor.codegen import wrapper, memory_planning
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from typing import Optional

from Simulator import functional_verify as _func_verify

from . import codecache, kernel_spec


def _writes(kernel_name, position):
    """Does `kernel_name` write the tensor argument at `position`?

    The roles are recorded at define_kernel. Unknown kernel or unknown
    position -> True, so an unrecorded argument stays checkable.
    """
    if kernel_name is None:
        return True
    return kernel_spec.writes_arg(kernel_name, position)


def _mutated(line):
    """Buffer names a NON-kernel wrapper line writes, from Inductor's own IR.

    A GENERATED KERNEL IS NOT THE ONLY THING THAT WRITES A BUFFER. Anything
    Inductor declines to codegen comes out as a fallback call that mutates its
    first argument in place, and the buffer is only finished after it:

        pytorchsim_triton_compiler_fused_eq_index_put_view_26(arg0_1, buf0, 1968)
        _fverify.verify_check(buf0, ...)                    <- was here
        aten.index_put_(buf0, [buf1], arg2_1, False)        <- finishes buf0

        measured   Kimi-VL's MoonViT merge, `inputs_embeds[input_ids ==
                   image_token] = image_features`. The kernel copies the
                   embeddings and the fallback scatters the image rows in, so
                   the check saw the copy and reported 492 of 1968 elements
                   over tol -- exactly the 4 image tokens x 123 hidden the
                   fallback had not written yet.

    `get_mutation_names` is Inductor's answer to the same question, so this
    asks it rather than pattern-matching the line class (there are four of
    them, and a fifth would be silently missed).
    """
    node = getattr(line, "node", None)
    if node is None:
        return ()
    try:
        return tuple(node.get_mutation_names())
    except Exception:  # noqa: BLE001 - a node type that does not answer
        return ()


class TritonNPUWrapperCodegen(wrapper.PythonWrapperCodegen):
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

    def wrap_kernel_call(self, name, call_args):
        """Render the call args before joining them.

        `generate` hands call_args straight to a join, so a sympy Integer is a
        TypeError. Every call passes here, which is what makes it one place.
        """
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

                # Configure logger for generated wrapper code
                _logger = setup_logger("PyTorchSimCodegen.generated_wrapper")

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = AsyncCompile()
                os.environ["TORCHSIM_LAST_COMPILED_MODULE"] = __file__
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )
        self.header.splice(
            """
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

        self._fverify_seen = set()
        self._fverify_last = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            # memory_plan_reuse() reaches self.estimate_peak through
            # AllocateLine.should_reuse_buffer, and upstream sets it in
            # run_wrapper_ir_passes -- which this override replaces, so nothing
            # else will. Missing it is not a planning miss but an AttributeError,
            # and only on a graph with a reuse candidate far enough back to need
            # the estimate: ResNet-18 hits it, add does not. Same guard upstream
            # uses, so buffer reuse off means no estimate to build.
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
        """{buffer name: id of the LAST kernel call that names it}.

        THE FIRST KERNEL TO NAME A BUFFER IS NOT ALWAYS THE ONE THAT FINISHES
        IT. One fx op can be split across several kernels, and then the buffer
        is only complete after the last of them -- checking after the first
        compares a half-built buffer against a finished golden and reports a
        divergence that is not one.

            measured   DeepSeek-V3's MoE router. `aten.scatter.value` comes out
                       as two kernels sharing one origin node:

                         pytorchsim_triton_compiler_fused_scatter_zeros_like_38(buf9, 256)
                         _fverify.verify_check(buf9, ...)        <- here
                         pytorchsim_triton_compiler_fused_scatter_zeros_like_39(buf8, buf9, 128)

                       38 writes the zeros and 39 scatters the ones, so the
                       check saw an all-zero buffer and reported "128/256
                       elements over tol, all npu=0 cpu=1" -- every one of the
                       scattered ones "missing". Running kernel 39 standalone
                       against a torch reference gives max_abs_err 0.

        So the walk is done twice: once to find where each buffer is last
        written, and once to emit. Same order, same buffers, one check each --
        only the position moves.
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
        """Emit per-kernel CPU verify calls for this kernel's output buffers.

        Each bare-identifier buffer arg the kernel WRITES is checked once,
        after the LAST kernel that writes it -- see _fverify_last_writer for
        why not the first. The buffer is mapped to its originating fx node (op)
        so the runtime check can compare against the CPU golden keyed by that
        node.

        WRITES, NOT NAMES. This used to check every bare-identifier argument,
        inputs included, and the docstrings on both halves said "writes" while
        the code said "names". The two part company under buffer REUSE: the
        wrapper renames storage (`buf20 = buf9  # reuse`), so one buffer's
        contents live under another buffer's name, and that name's
        `origin_node` describes what Inductor MEANT to put there. Check it
        after a kernel that only reads it and the comparison is against a value
        nothing computed.

            measured   Stable Diffusion v1.5's UNet. Two kernels take an
                       `in_out_ptr0` and never store to it; they are called
                       eight times between them, on buf20, buf88, buf104,
                       buf170, buf189, buf208, buf230 and buf298 -- and those
                       eight are EXACTLY the eight divergences the run
                       reported, each against an `add_N` node that no kernel
                       materialises. Nothing else in the model diverges; 217
                       kernels run between them.
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
                continue          # this kernel only reads it
            self._fverify_check_one(name, line_id)

    def _fverify_emit_mutation_checks(self, line):
        """The same check, after a FALLBACK that finishes a buffer in place.

        See `_mutated`: the last write to a buffer is not always a generated
        kernel, and a check emitted before the fallback compares a half-built
        buffer against a finished golden.
        """
        if self._fverify_last is None:
            self._fverify_last = self._fverify_last_writer()
        for name in _mutated(line):
            self._fverify_check_one(name, id(line))

    def _fverify_check_one(self, name, line_id):
        """One `verify_check` for `name`, if this line is its last writer."""
        if name in self._fverify_seen:
            return
        if line_id is not None and self._fverify_last.get(name) != line_id:
            return                # something later still writes this buffer
        self._fverify_seen.add(name)
        if name in V.graph.graph_inputs:
            return  # placeholders: golden == input, nothing to verify
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

    def memory_plan(self):
        self.lines = memory_planning.MemoryPlanner(self).plan(self.lines)
