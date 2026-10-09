"""The Python wrapper module Inductor generates around the compiled kernels.

Emits the header the wrapper needs (torchsim_compile,
the functional-verify calls) and walks the wrapper IR lines once.
"""
import contextlib
import dataclasses

import torch
from torch._inductor.codegen import wrapper
from torch._inductor.ir import GraphPartitionSignature
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


@dataclasses.dataclass
class _VerifyLine(wrapper.WrapperLine):
    """One per-kernel verify call in the wrapper body."""
    text: str

    def codegen(self, code):
        code.writeline(self.text)


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

    def write_header(self):
        """Inductor's own header, then the names the npu wrapper adds: torchsim_compile, verify, the log line."""
        super().write_header()
        self.header.splice(
            f"""
                from PyTorchSimFrontend.config import setup_logger
                from Simulator import functional_verify as _fverify
                from {codecache.__name__} import torchsim_compile

                _logger = setup_logger("PyTorchSimCodegen.generated_wrapper")
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )

    def write_args(self, input_names):
        """Unpack the inputs as Inductor does, then (per-kernel verify on) build the CPU golden from them."""
        super().write_args(input_names)
        if not _func_verify.enabled():
            return
        gm = getattr(V.graph, "module", None)
        if gm is not None:
            gid = _func_verify.register_graph(gm)
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
        if not _func_verify.enabled():
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
            lines.extend(_VerifyLine(c) for c in checks)
        self.lines = lines

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
        """The verify_check lines after this kernel, one per buffer it writes and finishes.

        Only written arguments are checked: a reused buffer that is only read carries another origin.
        """
        checks = []
        for pos, a in enumerate(call_args):
            if not isinstance(a, str):
                continue
            name = a.strip()
            if not name.isidentifier():
                continue
            if not _writes(kernel_name, pos):
                continue
            checks += self._fverify_check_one(name, line_id)
        return checks

    def _fverify_emit_mutation_checks(self, line):
        """The same checks, after a fallback that finishes a buffer in place."""
        checks = []
        for name in _mutated(line):
            checks += self._fverify_check_one(name, id(line))
        return checks

    def _fverify_check_one(self, name, line_id):
        """The `verify_check` line for `name` ([] or one), if this line is its last writer."""
        if name in self._fverify_seen:
            return []
        if line_id is not None and self._fverify_last.get(name) != line_id:
            return []
        self._fverify_seen.add(name)
        if name in V.graph.graph_inputs:
            return []
        try:
            buf = V.graph.get_buffer(name)
        except Exception:
            buf = None
        if buf is None:
            return []
        origin = getattr(buf, "origin_node", None)
        if origin is None:
            return []
        op = str(getattr(origin, "target", "?"))
        return [f'_fverify.verify_check({name}, "{name}", "{origin.name}", "{op}")']
