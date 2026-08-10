"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates themselves are not GPU-specific -- torch ships one
`triton_mm.py.jinja` for cuda, xpu, mtia and cpu -- but `use_triton_template`
gates on `is_gpu`, and GPU_TYPES is a hardcoded list with no registration hook.
"""

import os

import torch


def _register_npu_as_gpu():
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def _claim_triton_present():
    # has_triton() asks whether a supported *device* is available, not whether
    # triton is installed. The missing piece is a driver we never use.
    import torch.utils._triton as triton_utils
    import torch._inductor.scheduler as scheduler

    triton_utils.has_triton = lambda: True
    if hasattr(scheduler, "has_triton"):
        scheduler.has_triton = lambda: True


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
        # TODO: size these from the hardware config (lanes, spad per lane)
        # rather than taking the generic set.
        def __init__(self):
            super().__init__()
            self.exhaustive_configs = self.mm_configs

    # addmm and baddbmm carry a bias as input_nodes[0]; without their own entry
    # the mm heuristic is used with prefix_args=0 and def_kernel asserts.
    @register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        pass


#: The `groups` of the convolution being lowered right now, for the one question
#: that needs it and cannot reach it. Inductor picks a conv's block sizes from
#: `(m, n, k)` with `n` the weight's FIRST extent -- which for a grouped
#: convolution is every group's channels together, while the kernel it configures
#: indexes `GROUP_OUT_C = OUT_C // GROUPS`. Nothing in that call carries `groups`,
#: and the only frame that knows it is the lowering. See _clamp_conv_block_n and
#: _size_conv_blocks_from_the_machine, which are the writer and the reader.
_conv_groups = None


def _groups_now():
    return getattr(_conv_groups, "value", 1) or 1


def _clamp_conv_block_n():
    """Tell the conv heuristic how many channels a GROUP has.

    THE CLAMP EXISTS AND IT IS GIVEN THE WRONG NUMBER. `preprocess_mm_configs`
    already narrows BLOCK_N to the extent it is handed -- that is why a 32-channel
    convolution gets BLOCK_N 32 and not 128 -- and for a grouped convolution it is
    handed `out_chan`, all groups at once. So a depthwise layer, whose every group has
    ONE output channel, is configured with BLOCK_N = 128 and masks 127 of them
    away on every program.

        measured   mobilenet_v2: every depthwise convolution comes out
                   BLOCK_N=128 with GROUP_OUT_C=1, so 1/128 of each tile is live.
                   The DMA still moves the whole tile and add_spad still reserves
                   it.

    WRAPPED AT THE LOWERING because that is the only frame holding `groups`; the
    heuristic is called from inside it, so a thread-local set here is read there
    and cleared on the way out. Nested lowerings restore the previous value
    rather than assuming 1, since a convolution can be lowered while another is
    on the stack (conv1d converts to conv2d and re-enters).
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

    # EVERY KEY THE LOWERING IS UNDER, and there are three. `register_lowering`
    # files it under the OpOverloadPacket AND under each of its overloads, and a
    # lowered graph reaches for `aten.convolution.default` -- so wrapping the
    # packet alone fires on nothing. MEASURED: with only the packet wrapped,
    # mobilenet's depthwise layers still came out BLOCK_N=128.
    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_conv_blocks_from_the_machine():
    """Offer conv tiles this machine has lanes for.

    `get_config_heuristics` has no registry lookup -- it is an if/elif over
    cuda, xpu, cpu, mtia and then `BaseConfigHeuristic()` -- so npu takes the
    generic set, whose first entry is ConvConfig(64, 256, 16, 2, 4). With 128
    lanes that is a [64, 256] tile banked on the N axis, two columns per lane,
    and bank_vectorize refuses it:

      operand [64, 256] banked on axis 1 with strides [1, 64] is not reached by
      indexing its own row-major type in the nest [2, 64] this op's maps ask for

    resnet18 reaches it at its eleventh conv, the first whose out_chan is 256 --
    below that `preprocess_mm_configs` clamps BLOCK_N to the channel count and
    the tile happens to fit.

    THIS IS THE MAPPING POLICY, NOT A WORKAROUND FOR THAT REFUSAL. A block size
    is a statement about the machine the kernel runs on, and taking a table
    written for a GPU is not one -- the heuristic registered below has carried
    the same TODO since it was written. The refusal is a real defect and stays
    one: a tile deeper than one element per lane is a shape this backend has to
    handle, and pinning BLOCK_N to the lane count only stops resnet from being
    the thing that reports it.

    N IS THE LANE AXIS, so it takes the lane count exactly -- the same choice
    kernel_spec.fixed_config_for makes for XBLOCK, and for the same reason. M
    and K are per-lane depth and cost scratchpad rather than lanes, so they are
    offered small-to-large and the first that the shape does not clamp wins.
    """
    from torch._inductor.choices import InductorChoices
    from torch._inductor.template_heuristics.triton import (
        BaseConfigHeuristic, ConvConfig)

    from PyTorchSimFrontend import extension_config

    lanes = int(extension_config.vpu_num_lanes)

    class NPUConfigHeuristic(BaseConfigHeuristic):
        def __init__(self):
            super().__init__()
            self.conv_configs = [
                ConvConfig(64, lanes, 16, 1, 4),
                ConvConfig(32, lanes, 16, 1, 4),
                ConvConfig(64, lanes, 32, 1, 4),
            ]

        def get_conv_configs(self):
            # THE CHANNELS ONE PROGRAM INDEXES, which for a grouped convolution
            # is not the extent Inductor passes. See _clamp_conv_block_n.
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
    """Launch a grouped convolution over the channels a GROUP has, not all of them.

    Inductor's conv grid asks for the whole channel count on the axis its kernel
    indexes PER GROUP:

        def conv2d_grid(n, c, h, w, meta, *, cdiv):
            return (cdiv(n * h * w, meta["BLOCK_M"]),
                    cdiv(c, meta["BLOCK_N"]),     <-- c is OUT_C, all groups
                    meta["GROUPS"])

    and inside the template `idx_y_c = program_id(1) * BLOCK_N + arange(BLOCK_N)`
    is masked against `GROUP_OUT_C = OUT_C // GROUPS`. The two disagree for every
    grouped convolution: axis 1 runs `cdiv(OUT_C, BLOCK_N)` blocks and only
    `cdiv(GROUP_OUT_C, BLOCK_N)` of them have a live column. The rest are
    launched, DMA their tiles, mask everything away and write nothing.

        measured   mobilenet_v2 on this backend. Its depthwise convolutions have
                   GROUP_OUT_C = 1, so every one of them overshoots:

                     GROUPS=960  BLOCK_N=128  grid=(1, 8, 960)   7680 programs
                     GROUPS=576  BLOCK_N=128  grid=(4, 5, 576)  11520
                     GROUPS=384  BLOCK_N=128  grid=(4, 3, 384)   4608
                     GROUPS=144  BLOCK_N=128  grid=(49, 2, 144) 14112

                   In the first, blocks 1..7 of axis 1 hold `idx_y_c` >= 128
                   against a bound of 1 -- 6720 of 7680 programs do literally
                   nothing, and the simulator runs every one.

    IT IS A CORRECTNESS-PRESERVING FIX AND NOT A HEURISTIC. The programs removed
    are exactly those whose stores are masked off in full, so the output is the
    same tensor; what changes is how many times the machine is asked to produce
    nothing. GROUPS == 1 is untouched -- there `GROUP_OUT_C` IS `OUT_C` and the
    two expressions are the same number.

    PATCHED ON THE TEMPLATE, not on the module. `conv2d_template` captured the
    function at construction, so rebinding `conv.conv2d_grid` alone changes
    nothing that runs; the object's own attribute is what the launcher reads.
    """
    from torch._inductor.kernel import conv as conv_kernel
    # `SymbolicGridFn` lives in select_algorithm, which is where conv.py itself
    # imports it from; torch._inductor.ir does not re-export it.
    from torch._inductor.select_algorithm import SymbolicGridFn

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
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
    heuristic (cpu) still has a choice.

    TODO: rank by simulated cycles; timing.run_togsim already returns one per
    compiled kernel.
    """
    from torch._inductor.select_algorithm import ExternKernelCaller

    return {c: (1e3 if isinstance(c, ExternKernelCaller) else 1.0) + i * 1e-3
            for i, c in enumerate(choices)}


def _short_circuit_degenerate_gemms():
    """A zero-length axis has no tile, so the heuristics offer no config and the
    empty choice list raises. A MoE expert routing no tokens gives [0, K] @ [K, N].
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.lowering import full, lowerings
    from torch._inductor.virtualized import V

    def wrap(op, bias):
        def wrapped(*args, _orig=lowerings[op], **kwargs):
            try:
                m, n, k, layout = mm_args(*args[bias:bias + 2],
                                          layout=kwargs.get("layout"))[:4]
                m, n, k = (int(V.graph.sizevars.size_hint(s)) for s in (m, n, k))
            except Exception:  # noqa: BLE001 - dynamic shape; leave it to _orig
                return _orig(*args, **kwargs)
            # k == 0 sums nothing, so zeros -- except addmm/baddbmm, which are
            # then beta * bias.
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


def _install_selection():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        return pick_config(choices)

    # Precompiling builds every candidate for the current GPU; we need only the
    # chosen kernel's source.
    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.make_precompile_fn = lambda self, *a, **k: (lambda: None)


_installed = False


def install():
    """On by default; TORCHSIM_TRITON_TEMPLATES=0 opts out. Sending mm to aten
    simulates nothing, so a test that stops inside tnpu says more than one that
    passes without running the op.
    """
    global _installed
    if _installed or os.environ.get("TORCHSIM_TRITON_TEMPLATES", "1") == "0":
        return
    from torch._inductor import config

    _register_npu_as_gpu()
    _claim_triton_present()
    _register_template_heuristics()
    _size_conv_blocks_from_the_machine()
    _size_grouped_conv_grid_per_group()
    _clamp_conv_block_n()
    _short_circuit_degenerate_gemms()
    _install_selection()

    # Not max_autotune: that also turns on pointwise autotuning, which appends
    # a benchmark harness to every kernel module.
    config.max_autotune_gemm = True
    # These are global but the heuristics are registered for npu only, so ATEN
    # stays in the list to keep a cpu gemm in the same graph from having no
    # choice at all. pick_config ranks it last.
    config.max_autotune_gemm_backends = "ATEN,TRITON"
    config.max_autotune_conv_backends = "ATEN,TRITON"
    config.triton.autotune_at_compile_time = False
    # Epilogue-fusion benchmarking renders a benchmark-flavoured kernel whose
    # harness imports land indented in the real module.
    config.benchmark_epilogue_fusion = False
    _installed = True
