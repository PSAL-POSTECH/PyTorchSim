"""Inductor scheduling for the Triton route.

`TritonNPUScheduling` keeps all of Inductor's Triton codegen and changes only
what happens to the source afterwards: `triton_npu_compile(...)` instead of
`async_compile.triton(...)`.

  define_kernel   emit our compile call into the wrapper.
  kernel_type     a TritonKernel whose call site is a plain python call, since
                  the name binds to our callable, not a `.run(grid=...)`
                  launcher.
"""

from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from . import kernel_spec


class TritonNPUKernel(TritonKernel):
    """A TritonKernel launched as a plain call.

    Upstream emits `kernel.run(a, b, xnumel, grid=grid(xnumel), stream=...)`,
    where `grid` is resolved at RUNTIME by triton_heuristics from the autotuned
    XBLOCK. There is no autotuner and no stream here: the kernel name is bound to
    the callable `triton_npu_compile` returned, so the call is `kernel(a, b, n)`.

    That is also why the block sizes must be fixed at CODEGEN time -- see
    kernel_spec.fixed_config_for. A grid that is only known after autotuning
    cannot be written into a tnpu KernelSpec.
    """

    # **kwargs: Inductor keeps adding parameters here and none apply to this
    # route, so they are accepted and ignored rather than pinning a torch
    # release.
    def call_kernel(self, name: str, node=None, **kwargs):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)
        # add_numel_to_call_args appends sympy values; ExtensionWrapperCodegen
        # joins call args as plain strings, so render them here.
        call_args = [a if isinstance(a, str) else str(a) for a in call_args]
        # triton=False -> PythonWrapperCodegen emits `name(args...)`, the same
        # shape the MLIR route uses (mlir_common.py:627).
        wrapper.generate_kernel_call(name, call_args, triton=False)


class TritonNPUScheduling(TritonScheduling):
    kernel_type = TritonNPUKernel

    count = 0

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = get_fused_kernel_name(node_schedule, "original_aten")
        kernel_name = "_".join(
            x for x in ("triton_npu", fused_name, str(TritonNPUScheduling.count)) if x
        )
        TritonNPUScheduling.count += 1
        wrapper.src_to_kernel[src_code] = kernel_name

        # Upstream substitutes these inside define_kernel; the tnpu side parses
        # the source, so they must be resolved before it leaves here.
        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        meta = kernel_spec.collect_meta(kernel, kernel_name)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"triton_npu_compile('''{src_code}''',")
        compile_wrapper.writeline(f"    meta={meta!r},")
        compile_wrapper.writeline(f"    kernel_name={kernel_name!r})")

        origins = ", ".join(
            sorted({str(o) for n in node_schedule
                    for o in getattr(getattr(n, "node", None), "origins", ()) or ()})
        )
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(),
                              f"# origins: {origins}")
        return kernel_name
