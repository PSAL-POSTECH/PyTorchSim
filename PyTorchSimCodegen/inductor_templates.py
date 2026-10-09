"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates are not GPU-specific; `use_triton_template` is.
"""

import os

import torch

from PyTorchSimFrontend import config

logger = config.setup_logger()

_conv_groups = None

_NPU_CHOICES = None

_installed = False


def _register_npu_as_gpu():
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def _claim_triton_present():
    """has_triton() asks whether a supported DEVICE is available, not whether
    triton is installed. The missing piece is a driver we never use."""
    import torch._inductor.scheduler as scheduler
    import torch.utils._triton as triton_utils

    triton_utils.has_triton = lambda: True
    if hasattr(scheduler, "has_triton"):
        scheduler.has_triton = lambda: True


def _lowering_npu():
    """Whether the graph being lowered right now targets this device."""
    from torch._inductor.virtualized import V

    try:
        return V.graph.get_current_device_or_throw().type == "npu"
    except Exception:  # noqa: BLE001 - no graph in scope is not npu
        return False


def _is_npu(obj):
    """Whether a device, device string, tensor, layout or node is on this device."""
    dev = getattr(obj, "device", obj)
    dev = getattr(dev, "device", dev)
    if isinstance(dev, str):
        return dev.split(":")[0] == "npu"
    return getattr(dev, "type", None) == "npu"


def _gemm_tiles(m, n, k, dtype_size, out_size=None):
    """This machine's mm tiles for [m, k] @ [k, n], best first: develop's select_tile.

    `gemm_tile_candidates` (develop's gemm_combination_mapping) enumerates every tile that
    fits half the scratchpad -- the other half is the next work-item's double buffer -- and
    ranks them by the scratchpad they use. Torch's generic set is appended after.
    `out_size`: the bytes C is stored at when it is wider than the inputs (a fused cast) --
    develop counts that as C once more per fused node, and with a widening cast it is the same.
    """
    from torch._inductor.template_heuristics.triton import GemmConfig

    from .hardware import HardwareInfo

    size = int(dtype_size)
    extra = max(int(out_size or size) // size - 1, 0)
    tiles = HardwareInfo().gemm_tile_candidates(int(m), int(n), int(k), n_extra_node=extra,
                                                precision_bytes=size)
    return [GemmConfig(tm, tn, tk, 1, 4) for tm, tn, tk in tiles]


def _register_template_heuristics():
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.template_heuristics.registry import (
        register_template_heuristic)
    from torch._inductor.template_heuristics.triton import (
        AddMMConfigMixin, BaseConfigHeuristic, MMTemplateConfigMixin)

    @register_template_heuristic(mm_template.uid, "npu")
    @register_template_heuristic(bmm_template.uid, "npu")
    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        def _get_config_generator(self):
            """The hook the mixin documents for exactly this.

            It is the one place the shape is known, and a tile mapping blind to
            M, N and K is not one. _finalize_mm_configs still dedupes and clamps.
            """
            generic = super()._get_config_generator()

            def configs(m, n, k, **kwargs):
                from torch._inductor.virtualized import V
                try:
                    mnk = [int(V.graph.sizevars.size_hint(s)) for s in (m, n, k)]
                except Exception:
                    yield from generic(m, n, k, **kwargs)
                    return
                mapped = _gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                if not mapped:
                    logger.warning(
                        "[torchsim-compile] no mapped tile for %sx%sx%s is a legal "
                        "Triton block; falling back to the generic set", *mnk)
                yield from self._finalize_mm_configs(mapped)
                yield from generic(m, n, k, **kwargs)

            return configs

    @register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        pass


def _groups_now():
    return getattr(_conv_groups, "value", 1) or 1


def _clamp_conv_block_n():
    """Tell the conv heuristic how many channels a GROUP has.

    `preprocess_mm_configs` narrows BLOCK_N to the extent it is handed, which for
    a grouped conv is all groups at once: mobilenet got 128 against GROUP_OUT_C 1.
    """
    import functools
    import inspect
    import threading

    global _conv_groups
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel

    _conv_groups = threading.local()
    sig = inspect.signature(conv_kernel.convolution)

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            try:
                groups = sig.bind(*args, **kwargs).arguments.get("groups", 1)
            except TypeError:
                groups = 1
            prev = getattr(_conv_groups, "value", 1)
            _conv_groups.value = groups if isinstance(groups, int) else 1
            try:
                return inner(*args, **kwargs)
            finally:
                _conv_groups.value = prev

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _lower_transposed_conv_as_a_direct_one():
    """Give a transposed convolution to the Triton template, as a direct one."""
    import functools
    import inspect

    from torch._inductor import ir
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.lowering import lowerings as L
    from torch._inductor.virtualized import V

    aten = torch.ops.aten
    prims = torch.ops.prims
    sig = inspect.signature(conv_kernel.convolution)

    def _per_dim(seq, n):
        seq = list(seq)
        return seq * n if len(seq) == 1 else seq

    def _spread(t, lead, extents, factor):
        """Put factor-1 zeros between neighbouring elements of each dim."""
        if all(f == 1 for f in factor):
            return t
        n = len(extents)
        paired = list(lead)
        for extent in extents:
            paired += [extent, 1]
        y = L[aten.view](t, paired)
        pad = []
        for i in reversed(range(n)):
            pad += [0, factor[i] - 1, 0, 0]
        y = L[aten.constant_pad_nd](y, pad, 0.0)
        y = L[aten.view](y, list(lead) +
                         [extents[i] * factor[i] for i in range(n)])
        for i in range(n):
            if factor[i] != 1:
                y = L[aten.slice](y, len(lead) + i, 0,
                                  extents[i] * factor[i] - (factor[i] - 1))
        return y

    def _rewrite(x, weight, stride, padding, dilation, output_padding, groups):
        """Return the (x, weight, *params) of the equivalent direct conv."""
        ints = V.graph.sizevars.guard_int_seq
        batch, in_chan, *spatial = ints(x.get_size())
        _, out_per_group, *kernel = ints(weight.get_size())
        n_sp = len(kernel)
        out_chan = out_per_group * groups
        stride = _per_dim(stride, n_sp)
        padding = _per_dim(padding, n_sp)
        dilation = _per_dim(dilation, n_sp)
        output_padding = _per_dim(output_padding, n_sp)

        y = _spread(x, [batch, in_chan], spatial, stride)

        pad = []
        for i in reversed(range(n_sp)):
            lo = dilation[i] * (kernel[i] - 1) - padding[i]
            pad += [lo, lo + output_padding[i]]
        if any(p != 0 for p in pad):
            y = L[aten.constant_pad_nd](y, pad, 0.0)

        w = L[aten.view](weight, [groups, in_chan // groups,
                                  out_chan // groups] + kernel)
        w = L[aten.permute](w, [0, 2, 1] + [3 + i for i in range(n_sp)])
        w = L[aten.view](w, [out_chan, in_chan // groups] + kernel)
        if any(k != 1 for k in kernel):
            w = L[prims.rev.default](w, [2 + i for i in range(n_sp)])

        w = _spread(w, [out_chan, in_chan // groups], kernel, dilation)

        return y, w, [1] * n_sp, [0] * n_sp, [1] * n_sp, [0] * n_sp

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
            except TypeError:
                return inner(*args, **kwargs)

            x, weight = a["x"], a["weight"]
            if not a["transposed"] or ir.get_device_type(x) != "npu":
                return inner(*args, **kwargs)

            batchless = len(x.get_size()) == len(weight.get_size()) - 1
            if batchless:
                x = L[aten.expand](x, [1, *x.get_size()])

            y, w, stride, padding, dilation, output_padding = _rewrite(
                x, weight, a["stride"], a["padding"], a["dilation"],
                a["output_padding"], a["groups"])
            out = inner(y, w, a["bias"], stride, padding, dilation, False,
                        output_padding, a["groups"])
            return L[aten.squeeze](out, dim=0) if batchless else out

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_conv_blocks_from_the_machine():
    """Offer conv tiles this machine has lanes for.

    `get_config_heuristics` has no registry lookup, so npu took a GPU's table.
    BLOCK_N is the lane count; M and K cost scratchpad and are offered ascending.
    """
    from torch._inductor.choices import InductorChoices
    from torch._inductor.template_heuristics.triton import (
        BaseConfigHeuristic, ConvConfig)

    from PyTorchSimFrontend import config

    lanes = int(config.vpu_num_lanes)

    class NPUConfigHeuristic(BaseConfigHeuristic):
        def __init__(self):
            super().__init__()
            self.conv_configs = [
                ConvConfig(64, lanes, 16, 1, 4),
                ConvConfig(32, lanes, 16, 1, 4),
                ConvConfig(64, lanes, 32, 1, 4),
            ]

        def get_conv_configs(self):
            base = super().get_conv_configs()

            def per_group(m, n, k, **kwargs):
                groups = _groups_now()
                return base(m, max(1, n // groups) if groups > 1 else n, k,
                            **kwargs)

            return per_group

    original = InductorChoices.get_config_heuristics

    def get_config_heuristics(self, device_type="cuda"):
        if device_type == "npu":
            return NPUConfigHeuristic()
        return original(self, device_type)

    InductorChoices.get_config_heuristics = get_config_heuristics


def _size_grouped_conv_grid_per_group():
    """Launch a grouped convolution over the channels a GROUP has.

    Inductor grids cdiv(OUT_C, BLOCK_N) on an axis the kernel masks against
    GROUP_OUT_C; the programs dropped are those whose stores were fully masked.
    """
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    orig2d = conv_kernel.conv2d_grid
    orig3d = conv_kernel.conv3d_grid

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        if not _lowering_npu():
            return orig2d(n, c, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        if not _lowering_npu():
            return orig3d(n, c, d, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_grid = conv2d_grid
    conv_kernel.conv3d_grid = conv3d_grid
    for tmpl, fn in ((getattr(conv_kernel, "conv2d_template", None), conv2d_grid),
                     (getattr(conv_kernel, "conv3d_template", None), conv3d_grid)):
        if tmpl is not None:
            tmpl.grid = fn


def pick_config(choices):
    """Stand in for benchmarking: there is no device to time on, so the offered
    order wins. Extern ranks last, present only so a device with no registered
    heuristic (cpu) still has a choice."""
    from torch._inductor.select_algorithm import ExternKernelCaller

    return {c: (1e3 if isinstance(c, ExternKernelCaller) else 1.0) + i * 1e-3
            for i, c in enumerate(choices)}


def _short_circuit_degenerate_gemms():
    """A zero-length axis has no tile, so the heuristics offer no config and
    the empty choice list raises. A MoE expert routing no tokens gives
    [0, K] @ [K, N]."""
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.lowering import full, lowerings
    from torch._inductor.virtualized import V

    def wrap(op, bias):
        def wrapped(*args, _orig=lowerings[op], **kwargs):
            try:
                m, n, k, layout = mm_args(*args[bias:bias + 2],
                                          layout=kwargs.get("layout"))[:4]
                m, n, k = (int(V.graph.sizevars.size_hint(s)) for s in (m, n, k))
            except Exception:
                return _orig(*args, **kwargs)
            if not _is_npu(layout):
                return _orig(*args, **kwargs)
            if m == 0 or n == 0 or (k == 0 and not bias):
                return full(layout.size, 0, dtype=layout.dtype,
                            device=layout.device)
            return _orig(*args, **kwargs)

        return wrapped

    aten = torch.ops.aten
    for op, bias in ((aten.mm, 0), (aten.bmm, 0),
                     (aten.addmm, 1), (aten.baddbmm, 1)):
        for name in op.overloads():
            o = getattr(op, name)
            if o in lowerings:
                lowerings[o] = wrap(o, bias)


def _num_cores():
    """This machine's core count. The frontend has it and nobody read it."""
    from .hardware import HardwareInfo
    try:
        return max(1, int(HardwareInfo().num_cores))
    except Exception:
        return 1


def _npu_choices_class():
    global _NPU_CHOICES
    if _NPU_CHOICES is not None:
        return _NPU_CHOICES
    from torch._inductor.choices import InductorChoices

    from . import launch

    _DT_BITS = {torch.float64: 64, torch.float32: 32, torch.float16: 16,
                torch.bfloat16: 16, torch.int64: 64, torch.int32: 32,
                torch.int16: 16, torch.int8: 8, torch.uint8: 8, torch.bool: 8}

    class NPUChoices(InductorChoices):
        def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
            """Pin this machine's blocks INSIDE Inductor, the hook upstream
            documents for exactly this ("used to apply fixed configurations").

            It runs before the TritonKernel is built, so the source is generated
            knowing the block: measured, a numel the block divides loses its mask
            entirely (`xmask = xindex < xnumel` -> `tl.full(...)`, loads lose the
            mask operand), and one it does not keeps it.
            """
            kw = super().triton_kernel_kwargs(kernel_cls, features, groups,
                                              kernel_kwargs)
            if not _lowering_npu():
                return kw
            from torch._inductor.codegen.triton import FixedTritonConfig

            from . import launch
            #: THE NUMELS ARE IN `groups`, one dict per tiling ({'x': 1024, 'r0_': 1}).
            numels = {}
            for g in (groups or []):
                try:
                    items = dict(g).items()
                except Exception:
                    continue
                for prefix, n in items:
                    try:
                        numels[f"{prefix}numel"] = int(n)
                    except Exception:
                        pass
            if not numels:
                return kw
            #: WIDEST ELEMENT, the same question `launch._element_bits` asks of
            #: the spec's args -- asked here of the nodes, which is what exists.
            bits = []
            for nd in features.scheduler_nodes():
                try:
                    bits.append(_DT_BITS.get(nd.node.get_dtype(), 32))
                except Exception:
                    pass
            args = ([{"dtype": {64: "float64", 32: "float32", 16: "float16",
                                8: "int8"}[max(bits)]}] if bits else [])
            shim = type("K", (), {"inside_reduction": bool(features.is_reduction()),
                                  "numels": numels})()
            try:
                cfg = launch.fixed_config_for(shim, numels, args) or {}
            except Exception as e:                       # noqa: BLE001
                logger.info("[torchsim-compile] no fixed_config for this kernel: %s", e)
                return kw
            cfg = {k: int(v) for k, v in cfg.items()
                   if v and k.endswith("BLOCK")}
            if not cfg:
                return kw
            kw = dict(kw)
            kw["fixed_config"] = FixedTritonConfig(cfg)
            return kw

        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            base = InductorChoices.should_use_persistent_reduction(
                features, cooperative_reduction)
            if not base or not _lowering_npu():
                return base
            try:
                extent = int(features.reduction_numel)
            except (TypeError, ValueError):
                return base
            if extent < 1:
                return base
            persistent = 1 << (extent - 1).bit_length()
            return persistent <= launch.reduction_block_for(extent)

        @staticmethod
        def reduction_split_factor(device, reduction_numel_hint, numel_hint,
                                   inner_reduction):
            """Split a reduction only when a core would otherwise sit idle.

            Splitting costs a DRAM round trip for the partials and buys
            parallelism across work-items that run AT THE SAME TIME. Two things
            have to hold: the machine must have more than one core, and the
            parallel axis must not already fill them. Upstream writes the kernel
            that combines the partials, so nothing here has to.
            """
            if getattr(device, "type", device) != "npu":
                return InductorChoices.reduction_split_factor(
                    device, reduction_numel_hint, numel_hint, inner_reduction)
            #: THE DEVICE IS THE ARGUMENT HERE, and the guard above already
            #: settled it. `_lowering_npu()` asks V.graph, which has no current
            #: device yet during lowering -- it answers False and the whole
            #: branch never runs.
            cores = _num_cores()
            if cores > 1:
                want = InductorChoices.reduction_split_factor(
                    device, reduction_numel_hint, numel_hint, inner_reduction)
                #: ENOUGH PARALLEL WORK ALREADY. `numel_hint` is the output
                #: count, i.e. the work-items the parallel axis alone gives; if
                #: that reaches the cores, splitting adds traffic and no overlap.
                try:
                    if int(numel_hint) >= cores:
                        return 1
                except (TypeError, ValueError):
                    return 1
                #: AND NO MORE THAN THE CORES. Upstream sizes the split for a GPU
                #: whose blocks number in the hundreds; here every split past the
                #: core count is a partial written and read for nothing.
                split = min(int(want), cores)
                if split > 1:
                    logger.info(
                        "[torchsim-compile] splitting a reduction of %s elements %s ways "
                        "over %s cores (upstream wanted %s)",
                        reduction_numel_hint, split, cores, want)
                    return split
                return 1
            if os.environ.get("TORCHSIM_LOG_SPLIT"):
                would = InductorChoices.reduction_split_factor(
                    device, reduction_numel_hint, numel_hint, inner_reduction)
                if would != 1:
                    logger.info(
                        "[torchsim-compile] declined a %s-way split of %s elements "
                        "into %s outputs (inner=%s)",
                        would, reduction_numel_hint, numel_hint, inner_reduction)
            return 1

    NPUChoices.__module__ = __name__
    NPUChoices.__qualname__ = "NPUChoices"
    globals()["NPUChoices"] = NPUChoices
    _NPU_CHOICES = NPUChoices
    return NPUChoices


def _install_npu_choices():
    """Register the choices this hardware makes differently: persistence, splits."""
    from torch._inductor import config as inductor_config

    if inductor_config.inductor_choices_class is None:
        inductor_config.inductor_choices_class = _npu_choices_class()


def _lower_conv1d_as_conv2d():
    """Send a 1-D convolution through the 2-D template instead of to aten.

    Inductor ships ONE conv template and gates it on `ndim == 2`
    (`torch/_inductor/kernel/conv.py`), so a Conv1d has no Triton choice to
    append and falls to `extern_kernels.convolution` -- which on npu raises
    `convolution_overrideable not implemented`. That is what stops
    RecurrentGemma's Griffin block, whose temporal conv is an `nn.Conv1d`.

    The rewrite itself is torch's own: `conv1d_to_conv2d` sits in
    `torch/_inductor/decomposition.py` under the comment "intentionally not
    regiestered" [sic], fully written and simply never registered. It adds a
    length-1 spatial axis, calls conv2d, and squeezes it back -- exact, not an
    approximation. We register it rather than writing our own.

    Registering a decomposition is the supported extension point, unlike the
    two patches above it in `install()`, which exist only because GPU_TYPES and
    has_triton are hardcoded lists with no hook.

    Guards, in order of what each prevents:
      * device -- the decomposition table is global, and a cpu graph in the
        same process must keep the lowering Inductor chose for it.
      * len(stride) != 1 -- this is what makes it a 1-D conv, AND what stops
        the recursion: the conv2d we emit comes back through here with a 2-D
        stride and declines.
      * transposed / output_padding -- conv1d_to_conv2d models neither.
    """
    from torch._inductor.decomposition import conv1d_to_conv2d, register_decomposition

    aten = torch.ops.aten

    @register_decomposition([aten.convolution])
    def _npu_convolution(input, weight, bias, stride, padding, dilation,
                         transposed, output_padding, groups):
        if input.device.type != "npu":
            return NotImplemented
        if len(stride) != 1 or transposed:
            return NotImplemented
        if any(p != 0 for p in output_padding):
            return NotImplemented
        return conv1d_to_conv2d(input, weight, bias, stride, padding, dilation, groups)


def _install_selection():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    orig_benchmark = AlgorithmSelectorCache.__dict__["benchmark_choices"].__func__
    orig_precompile = AlgorithmSelectorCache.make_precompile_fn
    orig_example = AlgorithmSelectorCache.__dict__["generate_example_value"].__func__

    def _choices_are_npu(choices):
        return any(_is_npu(getattr(c, "layout", None)) for c in choices)

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        if not _choices_are_npu(choices):
            return orig_benchmark(cls, choices, autotune_args, is_collective)
        return pick_config(choices)

    def make_precompile_fn(self, choices, *args, **kwargs):
        if not _choices_are_npu(choices):
            return orig_precompile(self, choices, *args, **kwargs)
        return lambda: None

    def generate_example_value(size, stride, device, dtype, extra_size,
                               allocation_size=None):
        """An unread benchmark operand, allocated rather than sampled."""
        if not _is_npu(device):
            return orig_example(size, stride, device, dtype, extra_size,
                                allocation_size)
        shape = size if allocation_size is None else allocation_size
        needed = extra_size
        if all(s > 0 for s in shape):
            needed += sum((s - 1) * t for s, t in zip(shape, stride)) + 1
        view = torch.as_strided(
            torch.empty(needed, dtype=dtype, device=device), shape, stride)
        return view if tuple(shape) == tuple(size) else view.as_strided(size, stride)

    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.make_precompile_fn = make_precompile_fn
    AlgorithmSelectorCache.generate_example_value = staticmethod(
        generate_example_value)


def install():
    """On by default; TORCHSIM_TRITON_TEMPLATES=0 opts out.

    Sending mm to aten simulates nothing, so a test that stops inside the compiler says
    more than one that passes without running the op.
    """
    global _installed
    if _installed or os.environ.get("TORCHSIM_TRITON_TEMPLATES", "1") == "0":
        return
    from torch._inductor import config as inductor_config

    _register_npu_as_gpu()
    _claim_triton_present()
    _register_template_heuristics()
    _lower_transposed_conv_as_a_direct_one()
    _size_conv_blocks_from_the_machine()
    _size_grouped_conv_grid_per_group()
    _clamp_conv_block_n()
    _short_circuit_degenerate_gemms()
    _install_npu_choices()
    _lower_conv1d_as_conv2d()
    _install_selection()

    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.max_autotune_conv_backends = "ATEN,TRITON"
    inductor_config.triton.autotune_at_compile_time = False
    inductor_config.benchmark_epilogue_fusion = False

    _installed = True
