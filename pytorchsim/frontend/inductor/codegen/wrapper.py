"""npu's Inductor wrapper: the Python module Inductor generates per graph, its kernels compiled by
torchsim_compile and called as plain functions, each extern call noted, and (verify on) a CPU
cross-check after each buffer's last writer."""

import contextlib
from typing import Optional

from torch._inductor.codegen import wrapper
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V

from pytorchsim.frontend.inductor.codegen import kernel_metadata, verify
from pytorchsim.frontend.inductor.codegen.user_kernel import user_kernel_compile, user_kernel_meta


class TritonNPUWrapperCodegen(verify.VerifyLines, wrapper.PythonWrapperCodegen):
    """The npu wrapper: kernels are compiled by torchsim_compile and called as plain functions."""

    def __init__(self):
        """A model's own Triton kernels: the kernels define_kernel is collecting for the one being
        defined (None between), Inductor's metadata line of each, the name of each launch variant."""
        self._npu_user_kernel = None
        self._npu_user_notes = {}
        self._npu_user_variants = {}
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
        """Inductor's own header, then the names the npu wrapper adds: torchsim_compile, verify, the log line."""
        super().write_header()
        self.header.splice(
            f"""
                from pytorchsim.config import setup_logger
                from pytorchsim.frontend.inductor.codegen import verify as _fverify
                from pytorchsim.backend.codecache import torchsim_compile
                from pytorchsim.backend.launch_hooks import torchsim_extern

                _logger = setup_logger("pytorchsim.frontend.inductor.codegen.generated_wrapper")
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )

    def note_extern(self, op, python_kernel_name):
        """Before an extern call, a torchsim_extern line naming its op: the op Inductor's IR node
        holds (aten.sort.stable), else the Python callable it calls (extern_kernels.mm)."""
        call = python_kernel_name if op is None else str(op)
        self.writeline(f"torchsim_extern({call!r})")

    def generate_extern_kernel_out(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_extern_kernel_out(node)

    def generate_extern_kernel_alloc(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_extern_kernel_alloc(node)

    def generate_extern_kernel_multi_out(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_extern_kernel_multi_out(node)

    def generate_fallback_kernel(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_fallback_kernel(node)

    def generate_fallback_kernel_with_runtime_lookup(self, buf_name, python_kernel_name, get_args,
                                                     op_overload, raw_args, outputs):
        self.note_extern(op_overload, python_kernel_name)
        super().generate_fallback_kernel_with_runtime_lookup(
            buf_name, python_kernel_name, get_args, op_overload, raw_args, outputs)

    def generate_scatter_fallback(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_scatter_fallback(node)

    def generate_index_put_fallback(self, node):
        self.note_extern(node.op_overload, node.python_kernel_name)
        super().generate_index_put_fallback(node)

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

    def define_user_defined_triton_kernel(self, kernel, configs, kwargs, restore_value_args,
                                          reset_to_zero_args, grids, epilogue_fusion,
                                          launch_kwargs):
        """A Triton kernel a model registers (triton_op, wrap_triton) is compiled by torchsim_compile like
        Inductor's own, not as the GPU's async_compile.triton; a launch whose spans or grid differ gets a
        name and compile call of its own, as the spec bakes them in."""
        if len(grids) != 1 or configs:
            raise NotImplementedError(
                f"{kernel.__name__}: a user-defined Triton kernel autotuned over configs; the npu "
                f"launches one grid fixed at codegen")
        if epilogue_fusion is not None:
            raise NotImplementedError(
                f"{kernel.__name__}: an epilogue fused into a user-defined Triton kernel")
        self._npu_user_kernel = []
        try:
            name, triton_meta, inductor_meta, extra = super().define_user_defined_triton_kernel(
                kernel, configs, kwargs, restore_value_args, reset_to_zero_args, grids,
                epilogue_fusion, launch_kwargs)
            defined = self._npu_user_kernel
        finally:
            self._npu_user_kernel = None
        notes = self._npu_user_notes
        for _, metadata in defined:
            notes[name] = metadata
        meta = user_kernel_meta(kernel, kwargs, triton_meta, grids[0])
        variants = self._npu_user_variants
        key = (name, repr(meta))
        if key not in variants:
            count = sum(1 for n, _ in variants if n == name)
            variants[key] = name if count == 0 else f"{name}_npu{count}"
            kernel_metadata.record_roles(variants[key], meta)
            super().define_kernel(variants[key], user_kernel_compile(kernel, meta), notes[name])
        return variants[key], triton_meta, inductor_meta, extra

    def define_kernel(self, kernel_name, kernel_body, metadata=None, gpu=True,
                      cpp_definition=None):
        """Inductor's own, except the async_compile.triton of the user-defined Triton kernel
        being defined, which define_user_defined_triton_kernel replaces."""
        if self._npu_user_kernel is not None:
            self._npu_user_kernel.append((kernel_name, metadata))
            return None
        return super().define_kernel(kernel_name, kernel_body, metadata, gpu, cpp_definition)

    def _generate_kernel_call_helper(self, kernel_name, call_args, *, device=None, triton=True,
                                     **kwargs):
        """An npu kernel call is a plain function call with Triton's argument rendering, a 0-d input
        unwrapped for the kernel's device; grid, stream and autotune do not apply. A C++ kernel keeps
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
