"""The two Inductor classes registered for `npu`: scheduling and the wrapper.

Inductor writes the Triton source itself. Scheduling takes each kernel's source and puts a
`torchsim_compile` call for it in the wrapper; the wrapper class renders the header and the
kernel calls of the Python module Inductor generates per graph.
"""

import contextlib
from typing import Optional

from torch._inductor.codecache import PyCodeCache, get_hash
from torch._inductor.codegen import wrapper
from torch._inductor.codegen.common import BackendFeature, IndentedBuffer
from torch._inductor.codegen.triton import TritonKernel, TritonScheduling
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import Placeholder, get_fused_kernel_name
from torch._inductor.virtualized import V

from pytorchsim import config
from pytorchsim.codegen import kernel_metadata, verify

_logger = config.setup_logger()


class TritonNPUKernel(TritonKernel):
    """A TritonKernel launched as a plain call.

    Upstream emits `kernel.run(..., grid=..., stream=...)`, resolved at runtime.
    There is no autotuner here, which is why the blocks are fixed at codegen.
    """
    def __init__(self, tiling, *args, fixed_config=None, **kwargs):
        """Pin this machine's blocks here, where every kernel of this class is made: the one
        Inductor emits and the one its fusion benchmark times are then the same kernel."""
        if fixed_config is None:
            from pytorchsim.frontend.inductor_patches import pinned_blocks
            fixed_config = pinned_blocks(tiling, kwargs.get("features"))
        super().__init__(tiling, *args, fixed_config=fixed_config, **kwargs)

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
        """The source Inductor benchmarks a fusion with, without its benchmark harness (the
        compiler takes the kernel alone), and its metadata, read off the kernel object as
        define_kernel reads it, kept for benchmark_codegened_module by the path the source loads
        from."""
        src_code = super().generate_kernel_code_from_nodes(nodes, False, hint_override, **kwargs)
        meta = kernel_metadata.collect_meta(self._codegen_kernel, "triton_")
        self._benchmarked[PyCodeCache.write(src_code)[1]] = (src_code, meta)
        return src_code

    def benchmark_codegened_module(self, mod, n_spills_threshold=8, node_names=None,
                                   skip_perf_cache=False):
        """(TOGSim cycles, path) of a module generate_kernel_code_from_nodes wrote: Inductor
        compares these with the cycles select.time_choices gave the unfused choices."""
        from pytorchsim.driver import select

        got = self._benchmarked.get(mod.__file__)
        if got is None:
            return float("inf"), mod.__file__
        return select.time_module(*got, mod.__file__), mod.__file__

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


def _tile_source(node_schedule):
    """How a template kernel's config was chosen, one of select's sources; None for any other
    kernel. A recorded config is the pick without an epilogue, so built with one fused it is
    RECORDED_UNFUSED; a MultiTemplateBuffer is only made for choices timed in cycles."""
    from torch._inductor.ir import MultiTemplateBuffer

    from pytorchsim.driver import select

    nodes = [n for n in node_schedule if hasattr(n, "is_template")]
    template = next((n for n in nodes if n.is_template()), None)
    if template is None:
        return None
    if isinstance(template.node, MultiTemplateBuffer):
        return select.TIMED
    ktc = getattr(template.node, "annotations", {}).get("ktc")
    source = ktc.annotations.get(select.SOURCE) if ktc is not None else None
    if source == select.RECORDED and len(nodes) > 1:
        return select.RECORDED_UNFUSED
    return source or select.HEURISTIC


class TritonNPUWrapperCodegen(verify.VerifyLines, wrapper.PythonWrapperCodegen):
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
                from pytorchsim.config import setup_logger
                from pytorchsim.codegen import verify as _fverify
                from pytorchsim.driver.codecache import torchsim_compile

                _logger = setup_logger("pytorchsim.codegen.generated_wrapper")
                _logger.info(f'Wrapper Codegen Path = {{__file__}}')
            """
        )

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
        """A Triton kernel a model registers (torch.library.triton_op, wrap_triton) is compiled by
        torchsim_compile like Inductor's own. Inductor names it and builds its triton_meta; the
        async_compile.triton it would emit for a GPU is not written. Inductor's cache keys the name
        by triton_meta alone; the spec also bakes each launch's spans and grid, so a launch that
        differs in those gets a name and compile call of its own."""
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
        notes = self.__dict__.setdefault("_npu_user_notes", {})
        for _, metadata in defined:
            notes[name] = metadata
        meta = user_kernel_meta(kernel, kwargs, triton_meta, grids[0])
        variants = self.__dict__.setdefault("_npu_user_variants", {})
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
        if getattr(self, "_npu_user_kernel", None) is not None:
            self._npu_user_kernel.append((kernel_name, metadata))
            return None
        return super().define_kernel(kernel_name, kernel_body, metadata, gpu, cpp_definition)

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


def _user_kernel_writes(kwargs):
    """The formal arguments the launch writes, as torch's analysis of the kernel's TTIR found them
    for its IR node (UserDefinedTritonKernel.arg_accesses, every tensor when the analysis failed),
    or None, with a warning, when the node being generated is not the launch's."""
    from torch._inductor import ir
    node = getattr(getattr(V.graph.scheduler, "current_node", None), "node", None)
    if not isinstance(node, ir.UserDefinedTritonKernel) or node.kwargs is not kwargs:
        _logger.warning("a user-defined Triton kernel's IR node is not the scheduler's current "
                        "node; every pointer argument is taken as written (inout)")
        return None
    return {dep.name for dep in node.arg_accesses.read_writes.writes}


def user_kernel_meta(kernel, kwargs, triton_meta, grid):
    """The spec meta of one launch of a user-defined Triton kernel: the signature and constants
    Inductor's triton_meta, each pointer's dtype and storage span its IR argument, its role
    written (inout) where torch's mutation analysis says the launch writes it, read (in)
    elsewhere, every pointer inout when that analysis is not at hand."""
    from pytorchsim.contract import storage_span
    from pytorchsim.driver.codecache import user_kernel_id

    hint = V.graph.sizevars.optimization_hint
    signature = {str(k): str(v) for k, v in triton_meta["signature"].items()}
    writes = _user_kernel_writes(kwargs)
    args, call_params = [], []
    for arg, token in signature.items():
        if token == "constexpr":
            continue
        call_params.append(arg)
        if not token.startswith("*"):
            continue
        node = kwargs[arg]
        layout = node.get_layout()
        args.append({
            "name": arg,
            "role": "inout" if writes is None or arg in writes else "in",
            "buffer": node.get_name(),
            "dtype": kernel_metadata._DTYPE[token],
            "numel": storage_span([int(hint(s)) for s in layout.size],
                                  [int(hint(s)) for s in layout.stride]),
        })
    return {
        "kernel_name": kernel.__name__, "signature": signature,
        "constants": dict(triton_meta.get("constants") or {}),
        "args": args, "call_params": call_params, "numels": {},
        "inside_reduction": False, "fixed_config": {},
        "template_grid": ([int(hint(g)) for g in grid] + [1, 1, 1])[:3],
        "user_defined": True, "user_kernel": user_kernel_id(kernel),
    }


def user_kernel_compile(kernel, meta):
    """The torchsim_compile call for a user-defined Triton kernel, with no source text read back:
    the source is the kernel's own and its jit helpers' (Inductor's transitive closure), written
    as it is (meta["user_defined"] keeps the Inductor-source rewriters off it)."""
    from torch._inductor.codegen.wrapper import (
        user_defined_triton_kernel_transitive_closure_source_code)

    src = IndentedBuffer()
    src.splice("""
        import triton
        import triton.language as tl
    """, strip=True)
    src.writeline("@triton.jit")
    src.splice(user_defined_triton_kernel_transitive_closure_source_code(kernel), strip=True)
    return (f"torchsim_compile({src.getvalue()!r},\n    meta={meta!r},\n"
            f"    kernel_name={kernel.__name__!r})")
