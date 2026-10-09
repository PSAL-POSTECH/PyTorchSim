"""Inductor scheduling for the Triton route.

Keeps ALL of Inductor's Triton codegen and changes only what happens to the
generated source afterwards: upstream hands it to `async_compile.triton`, this
hands it to `torchsim_compile`. Two overrides, and nothing else.
"""

from torch._inductor.codecache import get_hash
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from . import kernel_spec


class TritonNPUKernel(TritonKernel):
    """A TritonKernel launched as a plain call.

    Upstream emits `kernel.run(..., grid=..., stream=...)`, resolved at runtime.
    There is no autotuner here, which is why the blocks are fixed at codegen.
    """
    def call_kernel(self, name: str, node=None, **kwargs):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)
        wrapper.generate_kernel_call(name, call_args, triton=False)


class TritonNPUScheduling(TritonScheduling):
    kernel_type = TritonNPUKernel

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

        meta = kernel_spec.collect_meta(kernel, kernel_name)
        kernel_spec.demote_unwritten_inout(meta, src_code)
        kernel_spec.record_roles(kernel_name, meta)

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
