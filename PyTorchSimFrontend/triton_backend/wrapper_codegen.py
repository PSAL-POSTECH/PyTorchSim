"""Wrapper codegen for the Triton route.

Reuses `ExtensionWrapperCodegen` wholesale -- device guards, buffer allocation,
the TOGSimulator plumbing and the SRAM plan hooks are all route-independent --
and adds the one import the generated module needs: `triton_npu_compile`, which
is to this route what `custom_async_compile` is to the MLIR one.
"""

from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen

from . import codecache


class TritonNPUWrapperCodegen(ExtensionWrapperCodegen):
    def wrap_kernel_call(self, name, call_args):
        """Render the call args before joining them.

        `ExtensionWrapperCodegen.generate` writes a KernelCallLine by handing
        its `call_args` straight to `wrap_kernel_call`, which `", ".join`s them
        -- so anything that is not already a string is a TypeError. Upstream's
        triton path never hits that because it renders through
        `prepare_triton_kernel_call` first; the `triton=False` call site this
        route uses does not.

        Two producers put non-strings in there. A pointwise kernel's numels
        arrive from `add_numel_to_call_args` as sympy Integers, and a template
        kernel (mm/conv) carries its own sizes the same way. Rendering here
        rather than at either producer is what makes it one place: every kernel
        call in this route passes through this method.
        """
        return super().wrap_kernel_call(
            name, self.prepare_triton_kernel_call(call_args))

    def write_header(self):
        super().write_header()
        self.header.splice(
            f"""
            from {codecache.__name__} import triton_npu_compile
            """
        )
