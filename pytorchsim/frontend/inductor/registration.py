"""Everything npu registers with Inductor, one section per registry it fills, in install()'s order:

    Dynamo's device interfaces    NPUDeviceInterface: what Inductor asks of the device
    GPU_TYPES                     npu as a Triton GPU (a patch: it has no hook)
    Inductor's device codegens    scheduling, wrapper, post-grad pass
    Inductor's pre-grad pass      SDPA on shared K/V heads, pregrad.py's (one slot for every device)
    Inductor's reinplace          a chain of view scatters in place (a patch: it has no hook), postgrad/'s
    device op overrides           NPUDeviceOpOverrides: the device code in a wrapper
    Inductor's choices class      NPUChoices, answered by autotune/
    template heuristics           the mm/bmm configs and the resident mm's, autotune/'s
    codegen heuristics            the configs of a generated pointwise or reduction kernel, autotune/'s
    PTB's Triton driver           the runner a launch goes to: Spike and TOGSim
    Inductor's benchmarker        what picks one of a kernel's compiled configs
    Inductor's decompositions     npu's mm and bmm as with coordinate descent off (a patch: it has no hook)
    Inductor's realize on reuse   a row value a broadcast reads again computed once (a patch: it has no hook)
    Inductor's node lowering      an int8 matmul's dequantized result stored, its epilogue (a patch: it has no hook)
    Inductor's template fusions   each pending one benchmarked, though two compile one source (a patch: it has no hook)
"""

import functools
import weakref
from textwrap import dedent

import torch
from torch._dynamo.device_interface import DeviceInterface
from torch._inductor.codegen.common import DeviceOpOverrides

from torch._inductor.choices import InductorChoices
from torch._inductor.lookup_table import LookupTableChoices

from pytorchsim import config
from pytorchsim.backend import template_record
from pytorchsim.frontend.inductor import is_lowering_npu
from pytorchsim.frontend.inductor.autotune import generated_kernel_blocks, template_kernel_configs

logger = config.setup_logger()


def install():
    """Register npu in every registry: the device interface first, since Inductor's Triton check
    (has_triton) asks it."""
    _register_device_interface()
    _register_triton_gpu()
    _register_device_codegen()
    _register_pre_grad_pass()
    _register_scatter_reinplace()
    _register_device_op_overrides()
    _register_choices()
    _register_template_heuristics()
    _register_codegen_heuristics()
    _register_launcher()
    _register_benchmarker()
    _register_decompositions()
    _register_realize_on_broadcast()
    _register_realize_dequant()
    _register_template_fusion_futures()


# Dynamo's device interfaces (torch._dynamo.device_interface)

def _register_device_interface():
    """Register NPUDeviceInterface as npu's DeviceInterface."""
    from torch._dynamo.device_interface import register_interface_for_device

    register_interface_for_device("npu", NPUDeviceInterface)


class NPUDeviceProperties:
    """The properties Inductor reads; every other one it reads by getattr with a default."""

    def __init__(self):
        self.multi_processor_count = int(config.CONFIG_MACHINE.vector_lanes)


class NPUDeviceInterface(DeviceInterface):
    """npu's DeviceInterface: Dynamo's registry, read by Inductor (properties, compute capability,
    availability, Triton and bf16 support), answered for the simulator."""

    class Worker:
        @staticmethod
        def set_device(device: int):
            torch.npu.set_device(device)

        @staticmethod
        def current_device() -> int:
            return torch.npu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> NPUDeviceProperties:
            """One set for every npu device: the simulated machine's."""
            return NPUDeviceProperties()

    class device:
        """`with interface.device(index):` -- the one npu device, so nothing is switched."""

        def __init__(self, device):
            self.idx = device

        def __enter__(self):
            return self.idx

        def __exit__(self, *exc):
            return False

    @staticmethod
    def current_device() -> int:
        return torch.npu.current_device()

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        torch.npu.set_device(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        """The device a kernel loads on is the one device: nothing to switch, the previous is the same."""
        return torch.npu.current_device()

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return torch.npu.current_device()

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        """The npu has no stream a kernel is launched on: 0, as the wrapper's get_raw_stream says."""
        return 0

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        pass

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int:
        """None of CUDA's; Triton's compile metadata carries it, and PTB's backend ignores it."""
        return 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        """Kernels for npu are Triton, compiled by PyTorchSim-Triton-Backend."""
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True


# npu as a Triton GPU: GPU_TYPES, a patch Inductor has no hook for

def _register_triton_gpu():
    """The patch; has_triton and the backend hash are upstream's."""
    _add_gpu_type()


def _add_gpu_type():
    """Add npu to GPU_TYPES (torch/_inductor/utils.py:107), a plain list is_gpu reads (:3443; use_triton_template
    :1936, scheduler fusion :1497): no registry takes a device there."""
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


# Inductor's device codegens (register_backend_for_device)

def _register_device_codegen():
    """Register npu's scheduling, wrapper codegen and device_custom_pass."""
    from torch._inductor.codegen.common import register_backend_for_device

    from pytorchsim.frontend.inductor.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
    from pytorchsim.frontend.inductor.postgrad import pytorchsim_post_grad_pass

    register_backend_for_device("npu", TritonNPUScheduling, TritonNPUWrapperCodegen,
                                device_custom_pass=pytorchsim_post_grad_pass)


# Inductor's pre-grad pass (config.pre_grad_custom_pass): one slot for every device, so ours joins the list

def _register_pre_grad_pass():
    """Add pytorchsim_pre_grad_pass to pre_grad_custom_pass, after any pass already there; it rewrites npu
    nodes only."""
    from torch._inductor import config as inductor_config
    from torch._inductor.custom_graph_pass import get_custom_graph_passes

    from pytorchsim.frontend.inductor.pregrad import pytorchsim_pre_grad_pass

    passes = get_custom_graph_passes(inductor_config.pre_grad_custom_pass)
    if pytorchsim_pre_grad_pass not in passes:
        inductor_config.pre_grad_custom_pass = [*passes, pytorchsim_pre_grad_pass]


# Inductor's reinplace (fx_passes.reinplace.inplaceable_ops): a patch, it has no hook

def _register_scatter_reinplace():
    """An npu view scatter onto another scatter's result is written in place (postgrad/scatter.py)."""
    from pytorchsim.frontend.inductor.postgrad.scatter import install_scatter_chain

    install_scatter_chain()


# Inductor's device op overrides (register_device_op_overrides)

def _register_device_op_overrides():
    """Register NPUDeviceOpOverrides for npu."""
    from torch._inductor.codegen.common import register_device_op_overrides

    register_device_op_overrides("npu", NPUDeviceOpOverrides())


class NPUDeviceOpOverrides(DeviceOpOverrides):
    """The device code in an npu wrapper: no raw stream, no device to set or sync, a null guard."""

    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx: int) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: int) -> str:
        """A null context: the caller writes `with {this}:`, where "pass" would not parse."""
        return "torch._ops.contextlib.nullcontext()"


# Inductor's choices class (config.inductor_choices_class)

def _register_choices():
    """NPUChoices as Inductor's choices, unless something else already is."""
    from torch._inductor import config as inductor_config

    if inductor_config.inductor_choices_class is None:
        inductor_config.inductor_choices_class = npu_choices_class()


class NPUChoices(LookupTableChoices):
    """Inductor's lookup-table choices whose npu answers come from autotune/ (template kernels' configs and
    recorded picks, generated kernels' reductions); every other device keeps Inductor's."""

    def uuid(self):
        """This class's part of Inductor's cache key: the machine and the picks it hands out."""
        return f"pytorchsim-npu-choices-5-{template_record.graph_digest()}"

    def _finalize_template_configs(self, template_choices, kernel_inputs, templates, op_name,
                                   kwarg_overrides=None):
        """An npu template's choices: the external mapping's tile, else a recorded pick (an
        autotune strategy's only), else the heuristic's tiles."""
        if getattr(kernel_inputs, "device_type", None) == "npu":
            external = self._external_choices(kernel_inputs, templates, op_name)
            if external:
                return external
            if not template_record.autotunes():
                return InductorChoices._finalize_template_configs(
                    self, template_choices, kernel_inputs, templates, op_name, kwarg_overrides)
        return super()._finalize_template_configs(
            template_choices, kernel_inputs, templates, op_name, kwarg_overrides)

    @staticmethod
    def _external_tile_for(kernel_inputs, templates):
        """The tile the external mapping states for this npu mm or bmm, or None."""
        from torch._inductor.kernel_inputs import MMKernelInputs

        if getattr(kernel_inputs, "device_type", None) != "npu" or \
                not isinstance(kernel_inputs, MMKernelInputs) or \
                not any(t.uid in template_record.EXTERNAL_TEMPLATES for t in templates):
            return None
        return template_record.external_tile(*kernel_inputs.mnk_hinted())

    def _external_choices(self, kernel_inputs, templates, op_name):
        """The choice built at the tile the external mapping states for this mm or bmm's
        shape, marked external; [] when it states none. A tile this machine cannot build is refused."""
        from pytorchsim.frontend import tiling

        tile = self._external_tile_for(kernel_inputs, templates)
        if tile is None:
            return []
        uids = [t.uid for t in templates if t.uid in template_record.EXTERNAL_TEMPLATES]
        m, n, k = kernel_inputs.mnk_hinted()
        legal = [(c.block_m, c.block_n, c.block_k)
                 for c in tiling.gemm_tiles(m, n, k, kernel_inputs.dtype().itemsize)]
        if tile not in legal:
            raise ValueError(
                f"{config.codegen_external_mapping_file}: {m}_{n}_{k} -> {tile} is no tile this "
                f"machine builds (each side a lane multiple times a power of two, A, B and C in "
                f"the spad budget); its tiles, largest first: {legal[:8]}")
        tm, tn, tk = tile
        entry = {"BLOCK_M": tm, "BLOCK_N": tn, "BLOCK_K": tk, "EVEN_K": k % tk == 0}
        ktcs = LookupTableChoices._create_lookup_choices(
            self, {uid: [entry] for uid in uids}, templates, kernel_inputs, op_name)
        for ktc in ktcs:
            ktc.annotations[template_record.SOURCE] = template_record.EXTERNAL
        logger.info("[select] %s %sx%sx%s: the external mapping's tile %s", op_name, m, n, k, tile)
        return ktcs

    def get_template_configs(self, kernel_inputs, templates, op_name, kwarg_overrides=None):
        """Inductor's choices of `templates`, an npu mm or addmm offered the resident mm alone where
        it holds (mm_resident.templates_for) and the external mapping states no tile for it, its
        config marked as planned."""
        from pytorchsim.frontend.inductor.autotune import mm_resident
        offered = templates if self._external_tile_for(kernel_inputs, templates) else \
            mm_resident.templates_for(kernel_inputs, templates, op_name)
        got = super().get_template_configs(kernel_inputs, offered, op_name, kwarg_overrides)
        for c in got:
            ktc = c.annotations.get("ktc")
            if ktc is not None and ktc.template is mm_resident.mm_resident_template:
                ktc.annotations.setdefault(template_record.SOURCE, template_record.PLANNED)
        return got

    def get_config_heuristics(self, device_type="cuda"):
        if device_type == "npu":
            return template_kernel_configs.conv_config_heuristic()
        return super().get_config_heuristics(device_type)

    def _get_lookup_table(self):
        """This machine's recorded picks; Inductor's own looks only on CUDA."""
        return template_record.lookup_table()

    @staticmethod
    def _get_device_key(device):
        if getattr(device, "type", None) == "npu":
            return template_record.device_key()
        return LookupTableChoices._get_device_key(device)

    def _create_lookup_choices(self, *args, **kwargs):
        """Inductor's choices from the recorded configs, each marked as recorded."""
        ktcs = super()._create_lookup_choices(*args, **kwargs)
        for ktc in ktcs:
            ktc.annotations[template_record.SOURCE] = template_record.RECORDED
        return ktcs

    @staticmethod
    def can_fuse_vertical(scheduler, node1, node2, shared_data_score):
        """Inductor's answer, but an npu int32 matmul's stored reader fuses into no node but a matmul: taken into
        its own reader it would leave the matmul writing its int32 accumulator out."""
        return (not _int32_reader_apart(node1, node2) and
                LookupTableChoices.can_fuse_vertical(scheduler, node1, node2, shared_data_score))

    @staticmethod
    def can_fuse_horizontal(scheduler, node1, node2, shared_data_score):
        """Inductor's answer, but an npu int32 matmul's stored reader fuses into no node but a matmul."""
        return (not _int32_reader_apart(node1, node2) and
                LookupTableChoices.can_fuse_horizontal(scheduler, node1, node2, shared_data_score))

    @staticmethod
    def should_use_persistent_reduction(features, cooperative_reduction):
        return generated_kernel_blocks.use_persistent_reduction(
            features,
            InductorChoices.should_use_persistent_reduction(features, cooperative_reduction))

    def reduction_split_factor(self, device, reduction_numel_hint, numel_hint, inner_reduction):
        upstream = functools.partial(super().reduction_split_factor, device,
                                     reduction_numel_hint, numel_hint, inner_reduction)
        if getattr(device, "type", device) != "npu":
            return upstream()
        return generated_kernel_blocks.reduction_split(reduction_numel_hint, numel_hint,
                                                       upstream)


def npu_choices_class():
    """NPUChoices, the class registered as Inductor's choices."""
    return NPUChoices


# Inductor's template heuristics (torch._inductor.heuristics.template.registry)

def _register_template_heuristics():
    """Register the mm/bmm heuristic and its addmm/baddbmm form, and the resident mm's two, for npu."""
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.heuristics.template.registry import register_template_heuristic

    from pytorchsim.frontend.inductor.autotune import mm_resident
    from pytorchsim.frontend.inductor.autotune.template_kernel_configs import (
        mm_config_heuristics, mm_resident_config_heuristic)

    mm, addmm = mm_config_heuristics()
    register_template_heuristic(mm_template.uid, "npu")(mm)
    register_template_heuristic(bmm_template.uid, "npu")(mm)
    register_template_heuristic(mm_template.uid, "npu", op_name="addmm")(addmm)
    register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")(addmm)
    resident, resident_addmm = mm_resident_config_heuristic()
    register_template_heuristic(mm_resident.mm_resident_template.uid, "npu", op_name="mm")(resident)
    register_template_heuristic(mm_resident.mm_resident_template.uid, "npu", op_name="addmm")(resident_addmm)


# Inductor's codegen heuristics (torch._inductor.heuristics.registry)

def _register_codegen_heuristics():
    """Register npu's pointwise and reduction config heuristics."""
    from torch._inductor.heuristics.registry import register_codegen_heuristic

    from pytorchsim.frontend.inductor.autotune import generated_kernel_configs

    register_codegen_heuristic("pointwise", "npu")(generated_kernel_configs.NPUPointwiseConfigs)
    register_codegen_heuristic("reduction", "npu")(generated_kernel_configs.NPUReductionConfigs)


# PTB's Triton driver (set_launcher)

def _register_launcher():
    """Register the runner a compiled kernel's launch goes to."""
    from pytorchsim.backend import launcher

    launcher.install()


# Inductor's benchmarker (torch._inductor.runtime.benchmarking.register_benchmarker)

def _register_benchmarker():
    """Register npu's benchmarker: the pick among a kernel's compiled configs."""
    from torch._inductor.runtime.benchmarking import register_benchmarker

    from pytorchsim.backend import config_pick

    register_benchmarker("npu", config_pick.benchmark, override=True)


# Inductor's decompositions (torch._inductor.decomposition.decompositions): a patch, the table has no device hook

def _register_decompositions():
    """npu's mm and bmm decomposed as with coordinate descent tuning off, which torchao's quantize_ turns on
    for the process: a 1-row matmul stays one for postgrad/'s gemm rewrites, not a multiply and a sum."""
    from torch._inductor import decomposition

    packets = (torch.ops.aten.mm, torch.ops.aten.bmm)
    for op, fn in list(decomposition.decompositions.items()):
        if getattr(op, "overloadpacket", None) in packets and not getattr(fn, "_npu_without_cdt", False):
            decomposition.decompositions[op] = _without_coordinate_descent(fn)
    decomposition.fast_random_decomps.cache_clear()


def _without_coordinate_descent(fn):
    """`fn`, traced with coordinate descent tuning off when its first operand is on npu."""
    from torch._inductor import config as inductor_config

    @functools.wraps(fn)
    def decompose(self, *args, **kwargs):
        if getattr(getattr(self, "device", None), "type", None) != "npu":
            return fn(self, *args, **kwargs)
        with inductor_config.patch(coordinate_descent_tuning=False):
            return fn(self, *args, **kwargs)

    decompose._npu_without_cdt = True
    return decompose


# Inductor's realize on reuse (ir.StorageBox.should_realize_on_reuse): a patch, it has no hook

_NOT_ARITHMETIC = frozenset({"load", "load_seed", "to_dtype", "to_dtype_bitcast", "constant", "index_expr",
                             "indirect_indexing"})


def _register_realize_on_broadcast():
    """A pointwise npu value of one element per row (last dimension 1) that a broadcast reads again is stored once
    when it does more than load and cast, as XLA computes a row's scale in a fusion of its own; any other value,
    and a bool, is computed where it is read, as before."""
    from torch._inductor import ir
    from torch._inductor.virtualized import V

    original = ir.StorageBox.should_realize_on_reuse
    if getattr(original, "_npu_broadcast", False):
        return

    @functools.wraps(original)
    def should_realize_on_reuse(self, users, *, graph_reuse=True):
        if original(self, users, graph_reuse=graph_reuse):
            return True
        if graph_reuse or users <= 1 or not isinstance(self.data, ir.Pointwise) or not _is_npu_node(self.data):
            return False
        size = self.get_size()
        return (len(size) > 0 and V.graph.sizevars.statically_known_equals(size[-1], 1)
                and self.num_reads() > 0 and self.get_dtype() != torch.bool
                and bool(set(self.data.inner_fn_opcount().used_ops) - _NOT_ARITHMETIC))

    should_realize_on_reuse._npu_broadcast = True
    ir.StorageBox.should_realize_on_reuse = should_realize_on_reuse


def _is_npu_node(node):
    """Whether an IR node computes on npu."""
    from . import is_npu
    return is_npu(node.get_device())


# Inductor's node lowering (GraphLowering.run_node): a patch, it has no hook

def _register_realize_dequant():
    """An int8 matmul's 16-bit dequantized npu result is stored, so it is the matmul's epilogue as XLA's convolution
    fusion ends in bf16: inlined into its reader (a row norm, SiLU(gate) * up, the next quantization) it left the
    matmul writing its int32 accumulator out and the reader reading it back."""
    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering

    original = GraphLowering.run_node
    if getattr(original, "_npu_dequant", False):
        return

    @functools.wraps(original)
    def run_node(self, n):
        result = original(self, n)
        if isinstance(result, ir.TensorBox) and n in _dequant_ends(n.graph) and _is_npu_node(result):
            result.realize()
        return result

    run_node._npu_dequant = True
    GraphLowering.run_node = run_node


_DEQUANT_ENDS = weakref.WeakKeyDictionary()


def _dequant_ends(graph):
    """The FX nodes ending each aten._int_mm's dequantization: along its only users while they cast, view, or scale
    or shift by a smaller operand, up to a cast back to float32, the first 16-bit float cast after the last scale."""
    if graph in _DEQUANT_ENDS:
        return _DEQUANT_ENDS[graph]
    aten, prims = torch.ops.aten, torch.ops.prims
    casts = {prims.convert_element_type.default, aten._to_copy.default}
    views = {aten.view.default, aten._unsafe_view.default, aten.reshape.default}
    scales = {aten.mul.Tensor, aten.add.Tensor}
    ends = set()
    for mm in graph.find_nodes(op="call_function", target=aten._int_mm.default):
        chain, node = [], mm
        while len(node.users) == 1:
            user = next(iter(node.users))
            if user.target in scales:
                other = [a for a in user.args if a is not node]
                if not other or (isinstance(other[0], torch.fx.Node) and _reach(other[0]) >= _reach(node)):
                    break
            elif user.target not in casts | views:
                break
            if user.target in casts and _dtype(user) == torch.float32 and any(_dtype(c).itemsize == 2 for c in chain):
                break
            chain.append(user)
            node = user
        last = max((i for i, u in enumerate(chain) if u.target in scales), default=None)
        if last is None:
            continue
        end = next((u for u in chain[last + 1:] if u.target in casts and _dtype(u) in (torch.bfloat16, torch.float16)),
                   chain[last] if _dtype(chain[last]) in (torch.bfloat16, torch.float16) else None)
        if end is not None:
            ends.add(end)
    _DEQUANT_ENDS[graph] = ends
    return ends


def _int32_reader_apart(node1, node2):
    """Whether one of two scheduler nodes, neither a template, reads an npu template's int32 result."""
    if node1.is_template() or node2.is_template():
        return False
    return _reads_int32_template(node1) or _reads_int32_template(node2)


def _reads_int32_template(node):
    """Whether a scheduler node reads the int32 result of an npu matmul template."""
    from torch._inductor import ir
    from torch._inductor.virtualized import V
    for name in node.read_writes.buffer_names() - node.get_buffer_names():
        buf = V.graph.try_get_buffer(name)
        if isinstance(buf, ir.TemplateBuffer) and buf.get_dtype() == torch.int32 and _is_npu_node(buf):
            return True
    return False


def _reach(node):
    """The distinct elements an FX node's value holds (a broadcast dimension counted once), 0 when it records none."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return 0
    return functools.reduce(lambda a, b: a * b, (n for n, st in zip(val.shape, val.stride()) if st != 0), 1)


def _dtype(node):
    """The dtype of an FX node's value, int8 when it records none."""
    val = node.meta.get("val")
    return val.dtype if isinstance(val, torch.Tensor) else torch.int8


# Inductor's template fusions (Scheduler._evaluate_pending_template_fusions): a patch, it has no hook

def _register_template_fusion_futures():
    """Every pending template fusion is benchmarked: one whose compile future an earlier one already holds (two
    templates whose fused kernels are one source, a W8A8 MLP's gate and up) is evaluated synchronously, since
    Inductor maps each future to one pending fusion and dropped the other unbenchmarked."""
    from torch._inductor.scheduler import Scheduler

    original = Scheduler._evaluate_pending_template_fusions
    if getattr(original, "_npu_futures", False):
        return

    @functools.wraps(original)
    def evaluate(self, candidates, fused_nodes):
        taken = set()
        for pending_fusions in candidates.values():
            for pending in pending_fusions:
                future = pending.future.future if pending.future is not None else None
                if future is None:
                    continue
                if future in taken:
                    pending.future = None
                else:
                    taken.add(future)
        return original(self, candidates, fused_nodes)

    evaluate._npu_futures = True
    Scheduler._evaluate_pending_template_fusions = evaluate
