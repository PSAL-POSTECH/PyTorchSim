"""The two Inductor classes registered for `npu`: scheduling and the wrapper.

Inductor writes the Triton source itself. Scheduling takes each kernel's source and puts a
`torchsim_compile` call for it in the wrapper; the wrapper class renders the header and the
kernel calls of the Python module Inductor generates per graph.
"""

import contextlib
from typing import Optional

from torch._inductor.codecache import get_hash
from torch._inductor.codegen import wrapper
from torch._inductor.codegen.common import BackendFeature, IndentedBuffer
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from pytorchsim.codegen import block_sizing, kernel_metadata, verify


class TritonNPUKernel(TritonKernel):
    """A TritonKernel launched as a plain call.

    Upstream emits `kernel.run(..., grid=..., stream=...)`, resolved at runtime.
    There is no autotuner here, which is why the blocks are fixed at codegen.
    """
    def call_kernel(self, name: str, node=None, **kwargs):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)
        wrapper.generate_kernel_call(name, call_args, triton=True)


class TritonNPUScheduling(TritonScheduling):
    kernel_type = TritonNPUKernel

    @classmethod
    def get_backend_features(cls, device):
        """Triton's features without FOREACH: a foreach op is one kernel per tensor here. The
        machine has no launch to amortise, and a combo kernel picks its tensors by program id,
        a branch the TOG cannot time."""
        return super().get_backend_features(device) - {BackendFeature.FOREACH}

    def define_kernel(self, src_code, node_schedule, kernel):
        """Name the kernel by its source, capture its metadata, and emit our compile call.

        The name is resolved into the source here, because the compiler side parses it.
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = get_fused_kernel_name(node_schedule, "original_aten")
        kernel_name = "_".join(
            x for x in ("pytorchsim_triton_compiler", fused_name, get_hash(src_code.strip())[1:9]) if x
        )
        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        meta = kernel_metadata.collect_meta(kernel, kernel_name)
        kernel_metadata.demote_unwritten_inout(meta, src_code)
        kernel_metadata.record_roles(kernel_name, meta)
        tiles = block_sizing.template_tiles(src_code, meta)
        if tiles:
            meta["template_tiles"] = [list(t) for t in tiles]

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"torchsim_compile('''{src_code}''',")
        compile_wrapper.writeline(f"    meta={meta!r},")
        compile_wrapper.writeline(f"    kernel_name={kernel_name!r})")

        origins = ", ".join(
            sorted({str(o) for n in node_schedule
                    for o in getattr(getattr(n, "node", None), "origins", ()) or ()})
        )
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(),
                              f"# origins: {origins}")
        return kernel_name


class TritonNPUWrapperCodegen(verify.VerifyLines, wrapper.PythonWrapperCodegen):
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

    def write_header(self):
        """Inductor's own header, then the names the npu wrapper adds: torchsim_compile, verify, the log line."""
        super().write_header()
        self.header.splice(
            f"""
                from pytorchsim.config import setup_logger
                from pytorchsim.codegen import verify as _fverify
                from pytorchsim.driver.codecache import torchsim_compile

                _logger = setup_logger("pytorchsim.codegen.generated_wrapper")
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )

    def write_args(self, input_names):
        """Unpack the inputs as Inductor does, then (per-kernel verify on) build the CPU golden from them."""
        super().write_args(input_names)
        if not verify.enabled():
            return
        gm = getattr(V.graph, "module", None)
        if gm is not None:
            gid = verify.register_graph(gm)
            self.prefix.writeline(
                f"_fverify.verify_init({gid}, [{', '.join(V.graph.graph_inputs.keys())}])")

    def _generate_kernel_call_helper(self, kernel_name, call_args, *, device=None, triton=True,
                                     **kwargs):
        """An npu kernel call is a plain function call with Triton's argument rendering (a 0-d
        input unwrapped to its value, decided for the kernel's device, since this line is rendered
        after the CPU island's); grid, stream and autotune do not apply. A C++ kernel keeps
        Inductor's own call."""
        if triton:
            with (V.graph.set_current_device(device) if device is not None
                  else contextlib.nullcontext()):
                call_args = self.prepare_triton_kernel_call(call_args)
        else:
            call_args = [a if isinstance(a, str) else str(a) if isinstance(a, (int, float, bool))
                         else wrapper.pexpr(V.graph.sizevars.simplify(a)) for a in call_args]
        self.writeline(self.wrap_kernel_call(kernel_name, call_args))

    def run_wrapper_ir_passes(self, is_inference):
        """Inductor's passes, then (per-kernel verify on) a check line after each buffer's last writer."""
        super().run_wrapper_ir_passes(is_inference)
        if not verify.enabled():
            return
        self._fverify_seen = set()
        self._fverify_last = self._fverify_last_writer()
        lines = []
        for line in self.lines:
            lines.append(line)
            if isinstance(line, wrapper.KernelCallLine):
                checks = self._fverify_emit_checks(line.call_args, id(line), line.kernel_name)
            elif isinstance(line, wrapper.WrapperLine):
                checks = self._fverify_emit_mutation_checks(line)
            else:
                checks = []
            lines.extend(verify.VerifyLine(c) for c in checks)
        self.lines = lines
