"""Inductor scheduling for the Triton route.

`TritonNPUScheduling` keeps ALL of Inductor's Triton codegen -- fusion, index
expressions, masking, reductions, the kernel source itself -- and changes only
what happens to the generated source afterwards. Upstream hands it to
`async_compile.triton(...)`, which calls `triton.compile` for a GPU; we hand it
to `triton_npu_compile(...)`, which runs the triton-npu pipeline for this NPU.

Two overrides, and nothing else:

  define_kernel   emit our compile call into the wrapper instead of upstream's.
  kernel_type     a TritonKernel whose call site is a plain python call, because
                  the name is bound to our callable rather than to a triton
                  launcher with a `.run(grid=..., stream=...)` interface.
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

    # **kwargs, not a fixed signature: Inductor keeps adding parameters here
    # (2.10 added `deallocate_ws`). None of them apply to this route -- there is
    # no triton launcher and no workspace to release -- so they are accepted and
    # ignored rather than pinning us to one torch release.
    def call_kernel(self, name: str, node=None, **kwargs):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)
        # The numels arrive here as SYMPY values; they are rendered in
        # TritonNPUWrapperCodegen.wrap_kernel_call, which is the one place every
        # kernel call in this route passes through -- a template kernel (mm,
        # conv) never reaches this method at all.
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

        # Upstream substitutes these two placeholders inside define_kernel; the
        # source still carries them here, and the tnpu side parses the source, so
        # they have to be resolved before it leaves this function.
        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        meta = kernel_spec.collect_meta(kernel, kernel_name)
        # AFTER the substitutions above, so the source measured is the source
        # that runs. Inductor's `inplace_buffers` says which buffer it MEANT to
        # write in place; this asks the kernel body what it actually stores.
        kernel_spec.demote_unwritten_inout(meta, src_code)
        kernel_spec.record_roles(kernel_name, meta)

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
