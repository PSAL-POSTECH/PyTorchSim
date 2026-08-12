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

    A CAP AT THE LANE COUNT, AND IT IS A DEFECT RATHER THAN A PROPERTY. This
    used to say no axis may exceed the lane count and call it a fact about the
    machine. It is not. Three separate things hid behind that sentence; two are
    fixed and the cap is now over the third.

    FIXED, THE WRAP. Measured on triton-npu, one thing at a time on the same
    512x512x512 kernel:

        wrap, 2x2 grid, (256, 256, 512)     110.93 off
        NO wrap, 2x2 grid, (256, 256, 512)  5.34e-05
        no wrap, one program, 256 cube      3.81e-05

    so neither the wide tile nor the grid was the fault, and the array
    instructions come out as the same 2 x 4 x 2 loop nest either way.
    `kernel_spec.clamp_instead_of_wrap` removes it.

    FIXED, THE BIAS. Running each suspect alone at BLOCK_N = 512, M = 32,
    K = 768, N = 1536:

        matmul(a, b)             6.96e-05
        matmul(a, bt.t())        5.53e-05
        addmm(bias, a, b)        5.84        <-- the bias
        addmm(bias, a, bt.t())   5.84

    An addmm epilogue loads its bias through a BROADCAST pointer tensor, and
    triton_shared's PtrAnalysis rewrote every op on the way to it except the
    broadcast, so the load found no descriptor at [BLOCK_M, BLOCK_N] and fell to
    a gather -- which then wrote each lane's single fetched value into both of
    the slots that lane owns. `PtrAnalysis::rewriteBroadcastOp` supplies the
    missing arm; the transfer comes out `dram_stride = [0, 1]` with no gather,
    and tests/ops/attention/test_gqa.py then passes with the cap lifted.

    NOT FIXED, AND NOT YET NAMED. With the cap off,
    tests/ops/fusion/test_prologue_fusion.py is 127.39 off. Its kernel is a bmm
    at BLOCK_M = BLOCK_N = 512, BLOCK_K = 64, one program per batch. A bare
    512x512x64 matmul at the SAME tile passes standalone on triton-npu at
    7.63e-06, so it is not the width -- it is the bmm form or the fusion, and
    saying which needs a measurement nobody has taken yet.

    So the cap stays, over one case instead of three. Every candidate it removes
    is a LARGER tile, so the cost is speed rather than reach.
    """
    from torch._inductor.template_heuristics.triton import GemmConfig

    from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo

    tiles = BaseMLIRHardwareInfo().gemm_combination_mapping(
        int(m), int(n), int(k), precision_bytes=int(dtype_size),
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
        # tiles that nothing here counts. This is a policy over an unknown, not
        # a bound -- an epilogue deeper than the half left for it still
        # overflows, and the real fix is the re-codegen loop the MLIR route has
        # (BaseMLIRKernel.recodegen, "spad overflow") and this route does not.
        budget_divisor=2,
        # No offline mapping study on this route -- see BaseMLIRHardwareInfo.
        dump_candidates=False)

    lanes = int(extension_config.vpu_num_lanes)
    out = []
    for tile_m, tile_n, tile_k in tiles:
        if not all(_power_of_two(b) and b >= _MIN_BLOCK
                   for b in (tile_m, tile_n, tile_k)):
            continue
        if max(tile_m, tile_n, tile_k) > lanes:
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

    original = InductorChoices.get_config_heuristics

    def get_config_heuristics(self, device_type="cuda"):
        if device_type == "npu":
            return NPUConfigHeuristic()
        return original(self, device_type)

    InductorChoices.get_config_heuristics = get_config_heuristics


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
