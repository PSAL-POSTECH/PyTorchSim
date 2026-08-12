"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates themselves are not GPU-specific -- torch ships one
`triton_mm.py.jinja` for cuda, xpu, mtia and cpu -- but `use_triton_template`
gates on `is_gpu`, and GPU_TYPES is a hardcoded list with no registration hook.
"""

import os

import torch

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()


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


#: Triton's smallest block: `tl.dot` refuses an operand shorter than this.
_MIN_BLOCK = 16


def _power_of_two(n):
    return n >= 1 and (n & (n - 1)) == 0


def _round_up_pow2(n):
    """The smallest legal Triton block extent that covers `n`."""
    v = _MIN_BLOCK
    while v < n:
        v *= 2
    return v


def _gemm_tiles(m, n, k, dtype_size):
    """This machine's mm tiles for [m, k] @ [k, n], best first.

    THE MAPPING IS PyTorchSim's OWN, not a table. `gemm_combination_mapping`
    enumerates every tile that is a multiple of the lane count and whose three
    operands fit half the scratchpad -- half because the tiles are double
    buffered -- checking the total AND the per-lane footprint, and ranks them by
    scratchpad used, descending. That is rule 6's enumerate-rank-select over the
    quantity that actually limits this machine, and it is the same call the MLIR
    route's gemm and bmm templates make, so the two routes now answer the tile
    question once (mlir_common.BaseMLIRHardwareInfo).

    WHAT IT REPLACES: torch's generic set, whose first entry is
    GemmConfig(64, 64, 16) for f32 -- sized for a GPU's shared memory and warp
    tiling, both of which this machine has neither of. Nothing in it knows the
    lane count or the scratchpad, so the tile it picked was right only by
    accident, and the TODO saying so had been in this file since it was written.

    ONE THING THE MLIR ROUTE DOES NOT NEED, AND IT IS TRITON'S: a block size
    reaches `tl.arange` and `tl.dot`, so it must be a power of two and at least
    16. `gemm_combination_mapping` pads to multiples of 8 or of the lane count
    and multiplies by divisors, so 384 and 104 are both reachable and neither is
    a legal Triton block. They are dropped rather than rounded -- rounding up
    breaks the scratchpad budget the mapping just proved, and rounding down
    hands back a tile nothing enumerated.

    The generic set is appended after them, never before: a shape whose every
    mapped tile is an illegal block size still has to compile, and `pick_config`
    takes the first offered, so a tail is a fallback rather than a competitor.

    NO CAP ON THE TILE SIZE. There was one, at the lane count, and this
    docstring called it a fact about the machine. It never was: it was two
    defects wearing one sentence, and it took five wrong guesses to get there.

    THE WRAP. Measured on triton-npu, one thing at a time on the same
    512x512x512 kernel:

        wrap, 2x2 grid, (256, 256, 512)     110.93 off
        NO wrap, 2x2 grid, (256, 256, 512)  5.34e-05
        no wrap, one program, 256 cube      3.81e-05

    so neither the wide tile nor the grid was ever the fault, and the array
    instructions come out as the same 2 x 4 x 2 loop nest either way.
    `kernel_spec.clamp_instead_of_wrap` removes it -- and reaching the BMM
    spelling of it, `tl.load(A)` against mm's `tl.load(A + (xindex))`, is what
    took tests/ops/fusion/test_prologue_fusion.py from 127.39 off to passing.

    THE BIAS. Running each suspect alone at BLOCK_N = 512, M = 32, K = 768,
    N = 1536:

        matmul(a, b)             6.96e-05
        matmul(a, bt.t())        5.53e-05
        addmm(bias, a, b)        5.84        <-- the bias
        addmm(bias, a, bt.t())   5.84

    An addmm epilogue loads its bias through a BROADCAST pointer tensor, and
    triton_shared's PtrAnalysis rewrote every op on the way to it except the
    broadcast, so the load found no descriptor at [BLOCK_M, BLOCK_N] and fell to
    a gather -- which then wrote each lane's one fetched value into both of the
    slots that lane owns. Fixed upstream by `PtrAnalysis::rewriteBroadcastOp`;
    the transfer comes out `dram_stride = [0, 1]` with no gather.

    Both reproducers live in triton-npu:
    kernels/coverage/tile/tile_bias_row_deeper_than_one_per_lane.py for the
    second, and tile_gemm_wide_tile_grid.py for a REAL gather at a lane depth
    above one -- which is still red, and which this route no longer reaches
    because it no longer makes gathers it does not need.
    """
    from torch._inductor.template_heuristics.triton import GemmConfig

    from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo

    # ASK ABOUT THE ROUNDED SHAPE, NOT THE REAL ONE. Every tile this mapping
    # returns is a divisor of the padded extent times the lane count, so a
    # power-of-two extent gives power-of-two tiles and nothing has to be
    # dropped. Asking about 100 gives 104 and asking about 384 gives 384, both
    # illegal blocks, and the whole shape then falls back to torch's table --
    # measured: 129x61x56 offered 2 tiles and kept 0, 100-cubed 1 and 0,
    # 384-cubed 8 and 1.
    #
    # ROUNDING UP IS SAFE BECAUSE THE TAILS ARE MASKED. M and N are bounded by
    # `kernel_spec.clamp_instead_of_wrap`, which is exactly what it is for, and
    # K by the template's own EVEN_K masks. The budget is computed on the
    # rounded shape, so it over-reserves rather than under-, and the
    # power-of-two filter below stays as the check that this held.
    tiles = BaseMLIRHardwareInfo().gemm_combination_mapping(
        _round_up_pow2(int(m)), _round_up_pow2(int(n)), _round_up_pow2(int(k)),
        precision_bytes=int(dtype_size),
        # The Triton grid IS the tile count, so the same reason the MLIR gemm
        # template asks for at least num_cores tiles applies here.
        min_tile=True,
        # HEADROOM FOR WHAT THIS ROUTE STAGES AND THIS MAPPING CANNOT SEE.
        # It budgets three tiles; Inductor fuses the epilogue into the same
        # kernel AFTER the config is chosen -- the scheduler decides that, and
        # this runs during autotune -- so the extra output-shaped tiles are not
        # countable here. The MLIR route has the number when it asks and passes
        # n_extra_node; this asks for two tiles' worth of slack in the mapping's
        # own vocabulary instead.
        #
        # IT USED TO BE THE i64 INDEX TILES, and that reason is gone:
        # kernel_spec.clamp_instead_of_wrap replaces `rm % M` with a load bound,
        # so no operand becomes an indirect transfer and no index tile is staged
        # at all. Measured on a ragged 100x100x100, whose every transfer now
        # reads `masked_axes = [0, 1], masked_fill = 0` and none reads
        # `indirect`. The slack stays for the epilogue.
        n_prologue_node=2, n_prologue_extra_read=2,
        # AND WHAT IT CANNOT COUNT. Inductor fuses the epilogue into this kernel
        # AFTER the config is chosen -- the scheduler decides it, and this runs
        # during autotune -- so the number of extra output-shaped tiles is not
        # knowable here. The MLIR route has the count when it asks and passes
        # n_extra_node; this one gives the mm's own staging half the
        # double-buffer budget and leaves the rest for whatever fuses in.
        #
        # WHY THERE IS A DIVISOR AT ALL: tests/ops/fusion/test_addmm_residual
        # at 512x512x512 fuses a bias AND a residual, two more output-shaped
        # tiles that nothing here counts.
        #
        # AND IT HOLDS, WHICH IS WHY THERE IS NO RETRY LOOP HERE. The MLIR route
        # re-codegens on a scratchpad overflow (BaseMLIRKernel.recodegen, "spad
        # overflow") and this route cannot, so the obvious next step was to
        # build one -- except nothing overflows. Measured, deliberately trying
        # to: a 512-cubed matmul with FIVE fused elementwise epilogue nodes
        # comes back at 1.98e-04, and a 1024-cubed one with TEN at 1.71e-03.
        # Both link. Halving the budget makes the mapping pick a smaller tile,
        # and a smaller tile makes the epilogue's own tiles smaller with it, so
        # the reservation scales with what it is reserving for.
        #
        # A retry path with no case that needs it is machinery nobody can check
        # (rule 2 wants a kernel that fails without the fix, and there is none),
        # so this stays a measured reservation rather than a stopgap. If a
        # kernel ever does overflow, THAT is the reproducer and the loop can be
        # built against it.
        budget_divisor=2,
        # No offline mapping study on this route -- see BaseMLIRHardwareInfo.
        dump_candidates=False)

    out = []
    for tile_m, tile_n, tile_k in tiles:
        if not all(_power_of_two(b) and b >= _MIN_BLOCK
                   for b in (tile_m, tile_n, tile_k)):
            continue
        out.append(GemmConfig(tile_m, tile_n, tile_k, 1, 4))
    return out


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

            It is the one place the shape is known -- the mixin calls what this
            returns as `configs(m, n, k, dtype_size=..., op_name=...)` -- and a
            tile mapping that does not see M, N and K is not a mapping. The
            configs still go through `_finalize_mm_configs`, so torch keeps
            doing the deduping and the num_warps clamp.
            """
            generic = super()._get_config_generator()

            def configs(m, n, k, **kwargs):
                from torch._inductor.virtualized import V
                try:
                    mnk = [int(V.graph.sizevars.size_hint(s)) for s in (m, n, k)]
                except Exception:  # noqa: BLE001 - unhinted dynamic shape
                    yield from generic(m, n, k, **kwargs)
                    return
                mapped = _gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                if not mapped:
                    logger.warning(
                        "[triton-npu] no mapped tile for %sx%sx%s is a legal "
                        "Triton block; falling back to the generic set", *mnk)
                yield from self._finalize_mm_configs(mapped)
                yield from generic(m, n, k, **kwargs)

            return configs

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
    the same TODO since it was written.

    N IS NOT KNOWN TO BE THE LANE AXIS, and this used to say it was. Nothing
    here picks the lane axis: tnpu's select_lane_axis does, from what the ops
    demand, and the answer is per-operand -- gemm and bmm carry two different
    ones on a single op. Measured over the tnpu dumps, 26 of 36 stamped kernels
    are axis 0 and gemm_fp16_kernel is axis 1 throughout, so neither letter is
    the rule.

    NOR IS ONE ELEMENT PER LANE REQUIRED, which was the other half of the claim.
    kernels/coverage/tile/tile_deeper_than_one_per_lane.py puts two per lane on
    axis 0 and comes out exact; tile_gemm_lane_axis_deeper.py does it on the
    axis a matmul demanded, at 9.54e-06 against a 1.53e-05 control, with nine
    memref<128x256xf32, 1> spad buffers to show the tile was really built.

    What is left is a size, not a layout: BLOCK_N takes the lane count because a
    tile should be at least as wide as the machine, and M and K cost scratchpad
    rather than lanes, so they are offered small-to-large and the first the
    shape does not clamp wins. The bank_vectorize refusal quoted above is still
    a real defect -- it is just not the reason for this number.
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


# WHAT WAS HERE, AND WHY IT IS NOT. `_persist_a_reduction_that_fits_one_tile`
# forced Inductor to call a reduction persistent whenever our block would cover
# it, so that it took the two-pass `welford_reduce_fallback` instead of emitting
# a three-tensor Welford this backend could not lower. The lowering exists now
# (triton_shared 46d70d7, ReduceGeneralConverter: seed from slice 0 along the
# reduced axis and fold the rest), so the premise is gone -- and forcing
# persistence was not free. It fixes the block at the next power of two above
# the extent, which is what put ViT's first LayerNorm at 97440 bytes/lane
# against a 65536 budget; letting Inductor loop instead brought that to 77504,
# and codecache's R0_BLOCK retry takes it the rest of the way.
#
# It was also a monkeypatch of `InductorChoices.should_use_persistent_reduction`
# -- a class attribute of someone else's class -- which is the thing rule 16
# forbids outright. Removing it removes that debt with it.

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



#: Built once and kept, because Inductor PICKLES the choices class into its FX
#: graph cache key. A class defined inside a function is a `<locals>` object and
#: pickling it raises, which does not fail the compile -- it silently disables
#: the cache and prints a traceback per graph. Measured:
#: "AttributeError: Can't pickle local object
#:  '_decline_persistence_we_cannot_resize.<locals>.NPUChoices'".
_NPU_CHOICES = None


def _npu_choices_class():
    global _NPU_CHOICES
    if _NPU_CHOICES is not None:
        return _NPU_CHOICES
    from torch._inductor.choices import InductorChoices

    from . import kernel_spec

    class NPUChoices(InductorChoices):
        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            base = InductorChoices.should_use_persistent_reduction(
                features, cooperative_reduction)
            if not base:
                return False
            try:
                extent = int(features.reduction_numel)
            except (TypeError, ValueError):
                # Dynamic. We cannot size a block for it either, so leave
                # Inductor's answer alone rather than guess in the dark.
                return base
            if extent < 1:
                return base
            persistent = 1 << (extent - 1).bit_length()
            return persistent <= kernel_spec.reduction_block_for(extent)

    NPUChoices.__module__ = __name__
    NPUChoices.__qualname__ = "NPUChoices"
    globals()["NPUChoices"] = NPUChoices      # picklable by name
    _NPU_CHOICES = NPUChoices
    return NPUChoices


def _decline_persistence_we_cannot_resize():
    """Persist a reduction only at a block this backend would have chosen.

    THE LEVER THE SCRATCHPAD RETRY PULLS IS THE BLOCK, and a persistent
    reduction takes it away. Inductor writes `R0_BLOCK: tl.constexpr = <next
    power of two above the extent>` INTO the generated source, so
    `kernel_spec.fixed_config_for`'s answer is never read and
    `codecache._shrink_reduction_blocks` halves a number the kernel does not
    consult.

    MEASURED, BERT-small kernel 0 (three embedding gathers + the first
    LayerNorm, r0_numel 768, 21 scratchpad globals). Inductor calls 768
    persistent -- it is under the INNER threshold of 1024 -- and bakes
    R0_BLOCK 1024. The retry then recompiles EIGHT TIMES, 128 -> 64 -> 32 ->
    16 -> 8 -> 4 -> 2 -> 1, and the measurement does not move by one byte:

        70944 bytes/lane over a budget of 65536, retrying with R0_BLOCK 128
        70944 ...                                                       64
        70944 ...                              32, 16, 8, 4, 2, 1

    then `blocks` is empty and it re-raises. With persistence declined the same
    kernel compiles on the FIRST try at our own block of 512.

    THE RULE IS THE ONE FACT, NOT A HEURISTIC: persist iff the persistent block
    IS the block we would have pinned. Equal, and nothing is given up -- the
    kernel gets the size we wanted and Inductor writes the two-pass form that
    needs no Welford. Bigger, and persisting trades our only correction for a
    tile we already know does not fit. BERT-tiny (extent 128, our block 128)
    keeps persistence and its numbers; BERT-small (extent 768, next power of
    two 1024, our block 512) loses it, which is the point.

    THIS IS THE DOCUMENTED HOOK, NOT A MONKEYPATCH -- rule 16.
    `torch._inductor.config.inductor_choices_class` names the class
    `virtualized._choices_default` instantiates, and virtualized.py says what it
    is for in as many words: "We virtualize InductorChoices to allow changing
    inductor heuristics from out of tree." An earlier version of this file
    forced persistence by REBINDING `InductorChoices.should_use_persistent_
    reduction`, and the comment that removed it called that out as the thing
    rule 16 forbids. The decision is the inverse of that one and the wiring is
    not the same wiring.
    """
    from torch._inductor import config

    # Only when nobody else has claimed it: this is a global, and stamping over
    # another out-of-tree backend's choices would be the same rudeness this
    # file just stopped committing against InductorChoices itself.
    if config.inductor_choices_class is None:
        config.inductor_choices_class = _npu_choices_class()


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
    _decline_persistence_we_cannot_resize()
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

    # SPLIT REDUCTIONS ARE BACK ON, on the condition the note that turned them
    # off wrote down for itself: "give `welford_combine` a lowering (or a
    # two-pass fallback of its own) FIRST, then turn this back on and measure."
    # The lowering landed in triton_shared 46d70d7 -- ReduceGeneralConverter
    # takes a body of any shape by seeding the accumulator from slice 0 along
    # the reduced axis and folding the rest -- so a split reduction's second
    # kernel, which combines partial (mean, m2, weight) triples, now compiles.
    #
    # The measurement that turned it off was convnextv2's
    # `convolution_native_layer_norm_permute_17`: 17 kernels compiled with the
    # split on, 35 with it off. Re-measure before trusting either number now;
    # the reason it produced an unbuildable kernel no longer holds.
    #
    # It stays worth revisiting for its OWN reason, which is unchanged: the
    # extra kernel is only a win where the two halves run at once, and tnpu
    # compiles one binary per kernel with the C wrapper walking the grid as a
    # sequential loop. That is the launcher, not the machine.

    _installed = True
