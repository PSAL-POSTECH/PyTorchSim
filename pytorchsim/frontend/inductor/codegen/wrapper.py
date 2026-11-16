"""npu's Inductor wrapper: the Python module Inductor generates per graph, its kernels compiled by
Inductor's async_compile.triton, each extern call noted, and (verify on) a CPU
cross-check after each buffer's last writer."""

from typing import Optional

from torch._inductor.codegen import wrapper
from torch._inductor.ir import GraphPartitionSignature, IRNode
from torch._inductor.virtualized import V

from pytorchsim.backend import compiler_input, launch_hooks
from pytorchsim.frontend.inductor.codegen import verify
from pytorchsim.frontend.inductor.codegen.user_kernel import user_kernel_spec


class TritonNPUWrapperCodegen(verify.VerifyLines, wrapper.PythonWrapperCodegen):
    """The npu wrapper: Inductor's, with a note before each extern call and the verify lines."""

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
        """Inductor's own header, then the names the npu wrapper adds: the launch notes, verify, the log line."""
        super().write_header()
        self.header.splice(
            f"""
                from pytorchsim.config import setup_logger
                from pytorchsim.frontend.inductor.codegen import verify as _fverify
                from pytorchsim.backend.launch_hooks import torchsim_call, torchsim_extern

                _logger = setup_logger("pytorchsim.frontend.inductor.codegen.generated_wrapper")
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )

    def note_extern(self, op, python_kernel_name):
        """Before an extern call, a torchsim_extern line naming its op: the op Inductor's IR node
        holds (aten.sort.stable), else the Python callable it calls (extern_kernels.mm)."""
        call = python_kernel_name if op is None else str(op)
        self.writeline(f"torchsim_extern({call!r})")

    def generate_kernel_call(self, kernel_name, call_args, *, triton=True, raw_args=None, **kwargs):
        """Before a Triton kernel's call, a torchsim_call line naming the buffer each argument is, which its
        launch records in launch.json."""
        if triton:
            self.writeline(f"torchsim_call({_get_arg_names(call_args, raw_args)!r})")
        super().generate_kernel_call(kernel_name, call_args, triton=triton, raw_args=raw_args, **kwargs)

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
        """A Triton kernel a model registers (triton_op, wrap_triton) is Inductor's own async_compile.triton
        call; this launch's spec joins its launch kwargs, which Inductor turns into the kernel's backend_options."""
        if epilogue_fusion is not None:
            raise NotImplementedError(
                f"{kernel.__name__}: an epilogue fused into a user-defined Triton kernel")
        options = compiler_input.backend_options(user_kernel_spec(kernel, kwargs, grids))
        got = super().define_user_defined_triton_kernel(
            kernel, configs, {**kwargs, **options}, restore_value_args, reset_to_zero_args, grids,
            epilogue_fusion, [*launch_kwargs, *options])
        launch_hooks.name_user_kernel(kernel.__name__, kernel)
        return got

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


def _get_arg_names(call_args, raw_args):
    """Per call argument, the buffer it is: its IR node's name (a ReinterpretView's is its buffer's), or the
    argument itself when the graph has a buffer or input of that name; None for a number, symbol or else."""
    raw = list(raw_args) if raw_args is not None and len(raw_args) == len(call_args) else [None] * len(call_args)
    return tuple(_get_buffer_name(a, r) for a, r in zip(call_args, raw))


def _get_buffer_name(arg, raw):
    if isinstance(raw, IRNode):
        try:
            return raw.get_name()
        except NotImplementedError:
            return None
    if isinstance(arg, str) and (arg in V.graph.graph_inputs or V.graph.try_get_buffer(arg) is not None):
        return arg
    return None
