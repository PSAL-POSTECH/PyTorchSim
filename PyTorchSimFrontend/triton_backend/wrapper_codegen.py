"""Wrapper codegen for the Triton route.

Reuses `ExtensionWrapperCodegen` wholesale -- device guards, buffer allocation,
the TOGSimulator plumbing and the SRAM plan hooks are all route-independent --
and adds the one import the generated module needs: `triton_npu_compile`, which
is to this route what `custom_async_compile` is to the MLIR one.
"""

from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen

from . import codecache


class TritonNPUWrapperCodegen(ExtensionWrapperCodegen):
    def write_header(self):
        super().write_header()
        self.header.splice(
            f"""
            from {codecache.__name__} import triton_npu_compile
            """
        )
