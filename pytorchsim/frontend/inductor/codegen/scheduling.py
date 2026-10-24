"""npu's Inductor scheduling: Triton's, emitting a torchsim_compile call for each kernel's source into
the wrapper, and timing on TOGSim the modules Inductor's epilogue-fusion benchmark writes."""

from torch._inductor.codecache import PyCodeCache, get_hash
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from pytorchsim import config
from pytorchsim.frontend.inductor.codegen import kernel_metadata

_logger = config.setup_logger()


class TritonNPUKernel(TritonKernel):
    """A TritonKernel launched as a plain call: upstream emits `kernel.run(..., grid=..., stream=...)`,
    resolved at runtime. Its blocks are fixed when it is made (NPUChoices.triton_kernel_kwargs)."""

    def call_kernel(self, name: str, node=None, **kwargs):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)
        wrapper.generate_kernel_call(name, call_args, triton=True)


class TritonNPUScheduling(TritonScheduling):
    """Triton's scheduling, emitting torchsim_compile calls, and timing on TOGSim the modules
    Inductor's epilogue-fusion benchmark writes."""
    kernel_type = TritonNPUKernel

    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._codegen_kernel = None
        self._benchmarked = {}

    @classmethod
    def get_backend_features(cls, device):
        """Triton's features without FOREACH: a foreach op is one kernel per tensor here. The
        machine has no launch to amortise, and a combo kernel picks its tensors by program id,
        a branch the TOG cannot time."""
        return super().get_backend_features(device) - {BackendFeature.FOREACH}

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        self._codegen_kernel = kernel
        return super().codegen_node_schedule_with_kernel(node_schedule, kernel)

    def _codegen_single_template(self, kernel, render, template_node, epilogue_nodes,
                                 prologue_nodes, *, only_gen_src_code=False):
        self._codegen_kernel = kernel
        return super()._codegen_single_template(kernel, render, template_node, epilogue_nodes,
                                                prologue_nodes, only_gen_src_code=only_gen_src_code)

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False, hint_override=None,
                                        **kwargs):
        """The source Inductor benchmarks a fusion with, without its harness, and its metadata read as
        define_kernel reads it, kept for benchmark_codegened_module by the path the source loads from."""
        src_code = super().generate_kernel_code_from_nodes(nodes, False, hint_override, **kwargs)
        meta = kernel_metadata.collect_meta(self._codegen_kernel, "triton_")
        self._benchmarked[PyCodeCache.write(src_code)[1]] = (src_code, meta)
        return src_code

    def benchmark_codegened_module(self, mod, n_spills_threshold=8, node_names=None,
                                   skip_perf_cache=False):
        """(TOGSim cycles, path) of a module generate_kernel_code_from_nodes wrote: Inductor
        compares these with the cycles template_timing.time_choices gave the unfused choices."""
        from pytorchsim.backend import template_timing

        got = self._benchmarked.get(mod.__file__)
        if got is None:
            _logger.warning("[torchsim-compile] %s was not written by generate_kernel_code_from_nodes: "
                            "its fusion is not timed, and loses", mod.__file__)
            return float("inf"), mod.__file__
        return template_timing.time_module(*got, mod.__file__), mod.__file__

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
        kernel_metadata.record_roles(kernel_name, meta)
        source = _tile_source(node_schedule)
        if source is not None:
            meta["tile_source"] = source

        origins = ", ".join(
            sorted({str(o) for n in node_schedule
                    for o in getattr(getattr(n, "node", None), "origins", ()) or ()})
        )
        wrapper.define_kernel(kernel_name, kernel_metadata.compile_call(src_code, meta, kernel_name),
                              f"# origins: {origins}")
        return kernel_name


def _tile_source(node_schedule):
    """How a template kernel's config was chosen, one of template_timing's sources; None for any other
    kernel. A recorded config is the pick without an epilogue, so built with one fused it is
    RECORDED_UNFUSED; a MultiTemplateBuffer is only made for choices timed in cycles."""
    from torch._inductor.ir import MultiTemplateBuffer

    from pytorchsim.backend import template_timing

    nodes = [n for n in node_schedule if hasattr(n, "is_template")]
    template = next((n for n in nodes if n.is_template()), None)
    if template is None:
        return None
    if isinstance(template.node, MultiTemplateBuffer):
        return template_timing.TIMED
    ktc = getattr(template.node, "annotations", {}).get("ktc")
    source = ktc.annotations.get(template_timing.SOURCE) if ktc is not None else None
    if source == template_timing.RECORDED and len(nodes) > 1:
        return template_timing.RECORDED_UNFUSED
    return source or template_timing.HEURISTIC
