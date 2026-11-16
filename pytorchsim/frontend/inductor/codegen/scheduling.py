"""npu's Inductor scheduling: Triton's, each kernel's source carrying its spec to PTB's Triton backend, and the
modules Inductor's epilogue-fusion benchmark writes timed on TOGSim."""

from torch._inductor.codecache import PyCodeCache, get_hash
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from pytorchsim import config
from pytorchsim.backend import compiler_input
from pytorchsim.frontend.inductor.autotune import generated_kernel_blocks, graph_prefetch
from pytorchsim.frontend.inductor.autotune.generated_kernel_configs import CANDIDATES_KEY
from pytorchsim.frontend.inductor.codegen import kernel_metadata

_logger = config.setup_logger()


class TritonNPUKernel(TritonKernel):
    """A TritonKernel whose heuristics' configs are this machine's candidate tiles (inductor_meta) and whose
    compile options carry its spec: the source is Inductor's own, autotune-ready, and each config that
    overflows the scratchpad is dropped when it is compiled."""

    def inductor_meta_per_kernel(self):
        out = super().inductor_meta_per_kernel()
        out[CANDIDATES_KEY] = self._get_candidates()
        return out

    def _get_candidates(self):
        """The candidate tiles, as dicts over the blocks this kernel takes as arguments, none twice."""
        taken = {f"{t.prefix.upper()}BLOCK" for t in self.range_trees
                 if t.tensor_dim is not None and not (t.is_reduction and self.persistent_reduction)}
        out = []
        for tile in generated_kernel_blocks.candidates_for(self.numels, self.features):
            tile = {k: v for k, v in tile.items() if k in taken}
            if self.no_x_dim:
                tile.pop("XBLOCK", None)
            if tile not in out:
                out.append(tile)
        return out or [{}]

    @property
    def triton_meta(self):
        return self.__dict__.get("_triton_meta")

    @triton_meta.setter
    def triton_meta(self, meta):
        """Inductor sets triton_meta once its signature is whole; the spec joins it there, before it renders."""
        self.__dict__["_triton_meta"] = meta
        if meta is not None:
            meta["backend_options"] = kernel_metadata.spec_options(self, str(Placeholder.KERNEL_NAME))


class TritonNPUScheduling(TritonScheduling):
    """Triton's scheduling over TritonNPUKernel, and the modules Inductor's epilogue-fusion benchmark writes
    timed on TOGSim."""
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
        kernel.triton_meta = kernel_metadata.TemplateTritonMeta(kernel)
        return super()._codegen_single_template(kernel, render, template_node, epilogue_nodes,
                                                prologue_nodes, only_gen_src_code=only_gen_src_code)

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False, hint_override=None,
                                        **kwargs):
        """The source Inductor benchmarks a fusion with, without its harness (triton.py:8308-8313 launches
        it once unmeasured, :8340-8349 subtracts a no-launch benchmark: -inf cycles), its metadata kept for
        benchmark_codegened_module by the path the source loads from."""
        src_code = super().generate_kernel_code_from_nodes(nodes, False, hint_override, **kwargs)
        kernel = self._codegen_kernel
        meta = kernel_metadata.collect_meta(kernel, "triton_")
        meta["graph_prefetch"] = graph_prefetch.statement(kernel, meta["args"])
        self._benchmarked[PyCodeCache.write(src_code)[1]] = (src_code, meta)
        return src_code

    def can_fuse_vertical(self, node1, node2):
        """Inductor's answer, and for a template fused with an epilogue or prologue, whether the fused
        kernel compiles: the choice was picked for the template alone, and a fusion adds tiles to its
        scratchpad (OutOfResources). A template still holding its timed choices is fused by benchmark."""
        if not super().can_fuse_vertical(node1, node2):
            return False
        if not (node1.is_template() or node2.is_template()):
            return True
        if any(isinstance(n.node, MultiTemplateBuffer) for n in (node1, node2) if n.is_template()):
            return True
        return self._compiles([*node1.get_nodes(), *node2.get_nodes()])

    def _compiles(self, nodes):
        """Whether the kernel fusing `nodes` compiles through the Triton backend with a config that fits."""
        from torch._inductor.runtime.triton_heuristics import NoTritonConfigsError
        from triton.runtime.errors import OutOfResources

        with V.graph.set_current_device(nodes[0].get_device()):
            mod = PyCodeCache.load(self.generate_kernel_code_from_nodes(nodes))
        try:
            mod.triton_.precompile()
        except (NoTritonConfigsError, OutOfResources) as e:
            _logger.info("[torchsim-compile] %s: not fused, its kernel does not fit: %s",
                         [n.get_name() for n in nodes], e)
            return False
        return True

    def benchmark_codegened_module(self, mod, n_spills_threshold=8, node_names=None,
                                   skip_perf_cache=False):
        """(TOGSim cycles, path) of a module generate_kernel_code_from_nodes wrote: Inductor
        compares these with the cycles the benchmarker gave the unfused choices."""
        from pytorchsim.backend import template_timing

        got = self._benchmarked.get(mod.__file__)
        if got is None:
            _logger.warning("[torchsim-compile] %s was not written by generate_kernel_code_from_nodes: "
                            "its fusion is not timed, and loses", mod.__file__)
            return float("inf"), mod.__file__
        return template_timing.time_module(*got, mod.__file__), mod.__file__

    def define_kernel(self, src_code, node_schedule, kernel):
        """Inductor's own: the kernel is an async_compile.triton call, compiled through the Triton backend; a
        template's source takes the spec of its final form first, and the roles of the arguments are kept by
        the kernel's name for the launch hooks."""
        meta = getattr(kernel, "_npu_meta", None)
        kernel_name = super().define_kernel(src_code, node_schedule, kernel)
        if meta is not None:
            kernel_metadata.record_roles(kernel_name, meta)
        return kernel_name


def _tile_source(node_schedule):
    """How a template kernel's config was chosen, one of template_timing's sources; None for any other
    kernel. A recorded config is the pick without an epilogue, so built with one fused it is
    RECORDED_UNFUSED; a MultiTemplateBuffer is only made for choices timed in cycles."""
    from torch._inductor.ir import MultiTemplateBuffer

    from pytorchsim.backend import template_record

    nodes = [n for n in node_schedule if hasattr(n, "is_template")]
    template = next((n for n in nodes if n.is_template()), None)
    if template is None:
        return None
    if isinstance(template.node, MultiTemplateBuffer):
        return template_record.TIMED
    ktc = getattr(template.node, "annotations", {}).get("ktc")
    source = ktc.annotations.get(template_record.SOURCE) if ktc is not None else None
    if source == template_record.RECORDED and len(nodes) > 1:
        return template_record.RECORDED_UNFUSED
    return source or template_record.HEURISTIC
