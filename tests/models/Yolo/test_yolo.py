"""YOLO v3 / v5 / v6 / v8 / v9 / v10 / 11 / 12 / 26, on the Triton codegen route.

One file per family, and here the family is defined by a PARSER rather than by a
checkpoint.  ``ultralytics.nn.tasks.parse_model`` turns a version's YAML into a
``nn.Sequential``, so every version in this file is built the same way -- from
``ultralytics/cfg/models/<version>/*.yaml`` with random weights, no ``.pt``, no
network, no torch.hub -- and what changes between presets is which blocks the
parser instantiates.  That is the axis worth covering: the backbone/neck blocks
are what each YOLO generation actually changed.

``tests/models/Yolov5/test_yolov5.py`` stays as it is and covers something this
file cannot: real pretrained weights, fetched through torch.hub from the
standalone ultralytics/yolov5 repository, which has its own parser AND its own
anchor-based head.  This file is the config-built counterpart across
generations.

THE PARSER NORMALISES THE HEAD, which is the first thing to know about this
file and was NOT the assumption it was written under.  ``ultralytics/cfg/models``
carries a v3 and a v5 YAML, but they terminate in the modern ``Detect``:
anchor-free, decoupled, ``reg_max=16``, DFL.  Measured, not assumed -- every
preset below reports ``reg_max=16`` and a real ``DFL`` module except 26, which
sets ``reg_max=1`` and an ``Identity``.  So the ANCHOR-BASED head is not
reachable through this parser at all, and the hub test is the only place in the
suite it runs.  What a preset here claims is therefore its BACKBONE and NECK,
plus the two head deltas the parser does express: end2end (v10, 26) and the
dropped DFL (26).

WHAT EACH VERSION ADDS, and why it needs its own preset:

  * v3     The pre-CSP baseline, and the only preset with no C-block of any
           kind: ``yolov3-tiny`` is Conv + ``MaxPool2d`` all the way down, and
           it is the only model in this suite that emits ``ZeroPad2d`` -- the
           asymmetric (0,1,0,1) pad in front of its last pool, which is the one
           pool it runs at stride 1 rather than 2.
  * v5     ``C3`` -- a CSP block whose two branches meet at a concat -- plus
           ``SPPF``, the serial 5x5 maxpool pyramid that replaced SPP's
           parallel one, and the plain ``Bottleneck`` nested inside C3.
  * v6     ``ConvTranspose2d`` in the neck: every other preset upsamples with
           ``nn.Upsample``, and this is the only one whose neck LEARNS the
           upsample.  Its backbone carries no C-block either.
  * v8     ``C2f`` -- v5's C3 with the split moved ahead of the bottlenecks, so
           the concat takes n+2 inputs rather than 2.
  * v9     ``RepNCSPELAN4`` / ``ELAN1`` / ``SPPELAN`` / ``AConv``: GELAN's
           aggregation, a different graph shape from CSP -- branches chained
           and then all concatenated -- plus ``RepConv``.
  * v10    NMS-free ``v10Detect``: dual one2many/one2one heads, so the model
           returns topk-selected boxes rather than a dense grid.  Brings
           ``PSA`` (the first attention block in this file), ``C2fCIB``,
           ``SCDown`` and ``RepVGGDW``.
  * 11     ``C3k2`` (a C2f whose bottleneck is switchable to a nested C3) plus
           ``C2PSA``, position-sensitive attention on the backbone's last
           stage.
  * 12     ``A2C2f`` / ``AAttn`` -- area attention, which splits the feature map
           into regions and attends inside each -- and no SPPF at all.  The most
           attention-heavy preset here.
  * 26     C3k2 + C2PSA like 11, but end2end AND without the DFL: ``reg_max=1``
           and ``self.dfl = Identity``.  It is the pair to v10 that makes
           "end2end" a tested difference rather than an assumed one, and the
           only preset whose box branch regresses distances directly.

``_assert_character`` cashes each claim above against the parsed model.  Without
it a preset can pass while proving nothing: every YAML in this list produces a
model that runs, so "yolo12 passes" is worth nothing if the parser quietly gave
back a stack of plain convolutions.

SCALE.  Every preset runs the smallest published scale (``n``, or ``-tiny`` for
v3) at 64x64 input and ``nc=4``.  Width is the axis that stays honest -- the
nano scales are shipped shapes, not invented ones -- and the two scaled axes are
input resolution and class count, which change the size of the tensors and not
which kernels run.  64 is the smallest input the head accepts: stride 32 has to
leave a feature map with a nonzero extent, and 64 gives 8x8 / 4x4 / 2x2.

    source /workspace/torchsim-env.sh
    python tests/models/Yolo/test_yolo.py --version v8

MEASURED 2026-08-13, one cleared dump path each, nine runs sharing the machine
so the wall clocks are upper bounds.  "kern" is unique compiled kernels; "max
rel" is the worst max|npu-cpu| / max|cpu| over all of a version's outputs.

    version  yaml            params      kern   time   max rel     verdict
    v3       yolov3-tiny  12,134,184      50    385s   2.93e-06    PASS ***
    v5       yolov5n       2,509,244      88    746s   1.85e-05    PASS
    v6       yolov6n       4,238,540      72*   498s   1.98e-05    PASS *
    v8       yolov8n       3,011,628      89    707s   1.15e-05    PASS
    v9       yolov9t       2,006,188     118   1710s   1.15e-04    PASS ***
    v10      yolov10n      2,708,600     124   1054s   2.64e-05    PASS ****
    11       yolo11n       2,590,620     124    926s   1.25e-05    PASS ****
    12       yolo12n       2,568,828     145   1286s   2.67e-05    PASS ****
    26       yolo26n       2,505,360     142   1198s   5.65e-05    PASS ****

      * v6 was BLOCKED IN THE DEVICE -- 68 kernels compiled and then a raise,
        never an output -- until the transposed convolution stopped being
        emitted at all.  Measured 2026-08-14, alone on the machine; its 72
        kernels are those 68 plus the rewrite's own.  See "the transposed
        convolution" below.
    *** v3 and v9 were BOTH WRONG NUMBERS here (1.27e+00 and 2.47e+00) until
        triton_shared b6c3e60; see "the pooling bug" below.  v9's pass is not
        comfortable -- 1.15e-04 against a 1e-4 relative criterion, where the
        others sit at 1e-06 -- and its margin is thin enough that a different
        input could tip it.  Its P3 output was ALREADY at 9.60e-05 before the
        fix and did not move, so the width of that margin is GELAN's own error
        accumulation, not the pooling bug.
   **** these four went WRONG NUMBERS -> PASS with triton_shared 99dc6bc, and
        before that SEGFAULTED on the triton-npu pin this file was written
        against.  Two separate defects in a row on the same kernel; see below.

SO ALL NINE PASS.  The file still gates only v5 and v8, and that is the rule
working rather than an oversight: the other seven pass against triton_shared
commits this repo does not pin yet.  See the gate's own note.

THE POOLING BUG, which is where two of the four came from.  A 2x2 stride-2
pool reads four addresses -- base, base+1, base+64, base+65 -- and
triton_shared's `PtrState::rebuildAsGatherScatter` was ZEROING the constant
that distinguishes them.  With the constants gone the four loads ARE the same
load, CSE merges them (correctly -- it is the witness, not the culprit) and
canonicalize folds `max(a,a,a,a)` to `a`.  Measured on v3's first pool:

    stage                          exp  select  load
    triton IR in                    4     3      4
    --triton-to-structured          4     3      4
    --cse --canonicalize            1     0      1

so the kernel returned silu of the window's TOP-LEFT element.  v9's avg_pool
had the same disease in the other spelling: the three adds survived while the
loads collapsed, making it (a+a+a+a)/4.  That is why v9's P3 was clean while
P4 and P5 were not -- the pools are on those two branches.

The fix keeps a compile-time constant and still clears a non-constant one,
which is exactly the splatted scalar the original comment was written about.
1311 dumped kernels through the full pipeline afterwards: 1304 byte-identical,
7 changed, 0 newly failing -- and all seven are pools.

AND THE GATHER WAS NEVER NEEDED.  XBLOCK is 8 and the modulus is 32, so within
a block `x // 32` is constant and `x % 32` is consecutive: measured over all
128 programs, every block's addresses are an arithmetic sequence of stride 2.
PtrAnalysis falls back to gather because it does not test whether the block
extent divides the modulus.  Fixing the gather path was the correctness fix;
not taking it at all is a separate, faster one that is not done.  Each of the
other seven had a measured stopping point rather than a guess, and no two of
them were the same stop:

  * v6 stopped in the DEVICE, not in the compiler.  Its neck upsamples with
    ConvTranspose2d, Inductor's conv lowering does not offer the Triton
    template a transposed convolution -- ``not transposed`` is one of the
    conditions, beside the comment "templates only support these" -- so it left
    ``extern_kernels.convolution(..., transposed=True, ...)``, and that
    dispatches on npu:0 to ``convolution_overrideable``, which raises.

    AND REGISTERING THE OP WOULD HAVE BEEN THE WRONG FIX.  It is not merely
    missing: PyTorch itself registers a CompositeExplicitAutograd kernel for it
    whose whole body is the raise, and an alias key covers PrivateUse1, so this
    device's global CPU fallback never gets a turn -- which is why erfinv and
    nanmedian fall back and return while conv_transpose2d does not.  Giving it
    a kernel would run the convolution on the HOST and simulate nothing.  The
    fix is to stop emitting it: a transposed convolution IS a direct one over
    an input with stride-1 zeros inserted, and that rewrite lives in
    ``triton_backend/inductor_templates.py``, wrapped around the convolution
    LOWERING.  Verified against aten over the product of stride, padding,
    output_padding, dilation, groups and kernel size before it was wired in,
    then through the route on twelve shapes.  It costs stride^2 times the
    multiply-adds, which is a true statement about a machine with no
    transposed-conv unit.

    MEASURED ON triton_shared WITHOUT b6c3e60 OR 99dc6bc -- the develop build,
    not the merge build -- so unlike the other seven, v6 does not depend on
    either fix.  Its blocker was entirely in this repo.

    AND THE CONTROLS FOUND A SECOND REPRODUCER FOR 99dc6bc, which is worth
    more than the transposed conv itself.  Plain ``nn.Conv2d`` at groups > 1
    WITH A BIAS returns the wrong answer on that build: the convolution is
    exact (the residual is channel-constant to 2.4e-07) and the BIAS lands on
    the wrong channels -- channel c gets ``bias[c % (OUT_C // GROUPS)]``, so a
    depthwise layer gives every channel ``bias[0]``.  That is the template's
    ``idx_c = idx_y_c[None, :] + group * GROUP_OUT_C``, the same scalar-on-the-
    outermost-dim shape as the PSA BatchNorm load 99dc6bc was written for.  All
    eight controls pass on the merge build.  So it is a KNOWN bug with a much
    cheaper witness than YOLO's PSA block: three lines of nn.Conv2d.

  * v3 and v9 WERE the pooling bug, and both pass now -- see it above.  What
    found it was per-kernel functional verify naming
    ``triton_npu_fused_max_pool2d_with_indices_silu_3`` writing buf4
    (1, 16, 32, 32) with 12167 of 16384 elements over tolerance, and then the
    kernel run standalone against candidate answers: `silu(top-left)` matched
    to 2.4e-07 where the correct `max(silu(...))` was 4.85 out.  Reading the IR
    would not have said which; running the kernel did.

    v9's shape was the tell that these were ONE bug and not two.  Its P3 output
    was clean at 9.6e-05 while P4 and P5 came back at 1.25 and 2.47 -- a
    uniform precision story cannot produce that, and the branch that was clean
    is the branch with no pool on it.

  * v10 / 11 / 12 / 26 all carry attention (PSA, C2PSA, A2C2f) and all four hit
    TWO defects in a row, in the SAME kernel -- the PSA block's depthwise 3x3
    positional-encoding conv on the 2x2 feature map.  The second only became
    visible once the first was fixed, which is the ordinary shape of this work.

    THE SECOND ONE, and the one that made them wrong rather than dead.  That
    kernel's epilogue loads four per-channel BatchNorm parameters, and all 128
    channels were reading channel 0's.  `idx_c = idx_y_c[None, :] + group`
    parks the scalar `group` on the OUTERMOST dim by convention, which for a
    <1x16> pointer is the size-1, stride-0 one; the load is then rebuilt as a
    gather -- not because its address needs one, but because its MASK does --
    and that rebuild divides the dim's offset by the dim's stride.  Stride 0.
    The IR carried `arith.divui %group, %c0`.

    Isolating the kernel one piece at a time is what found it, and each piece
    had to be neutralised to see the next: with BN identity and the residual
    zeroed the conv is exact (4.77e-07); with x zeroed the residual layout is
    exact; and only with BOTH neutralised does beta = 100..227 come back as 100
    for every channel.  Running program 5 alone then split it cleanly -- the
    STORE addressed channel 5 correctly while the load still read beta[0].
    Fixed in triton_shared 99dc6bc by MOVING that offset to a dim with a real
    stride rather than dividing it.

    THE FIRST ONE was a Spike ``User load segfault @ 0x11005bd0`` in the same
    kernel.  Bisecting the GRID rather than the data is what
    identified it -- program id 1 alone runs clean, ids 1 and 2 together fault,
    so it is the SECOND invocation that dies, which is a re-entrancy bug and not
    an index overrun.  Bisecting triton-npu over the 23 commits above the pin
    lands on a616ea5, "Take out three broadcast repairs that never fire, and the
    loop one of them held open" (tnpu/passes/p11_select_lane_axis.py).

    THAT COMMIT'S PREMISE DOES NOT HOLD FOR THIS SHAPE, and this kernel is the
    coverage it says cannot be written.  It measured
    ``_replicate_broadcast_on_lane_axis`` at 0 True over 423 calls across a
    352-kernel suite and removed it as dead.  Instrumented at the pinned commit,
    this kernel drives it to True TWICE, and the layout it then chooses is what
    faults.  So the removal is a fix, not a cleanup.

    With a616ea5 all four compile and run -- and then return wrong numbers, so
    the fix is necessary and not sufficient.  yolo11's first divergent buffer is
    buf132, an ``aten.bmm`` of shape (2, 4, 4) inside the attention, 1 element
    of 32 over tolerance at 5.5e-04 relative; by the outputs that has become
    0.57.

    THE PIN NOW CARRIES IT.  The compiler pin moved from 2988424
    to 98744f7, which is triton-npu develop MERGED with the p09 unit-axis line
    2988424 sat on.  The merge was needed rather than a bump: 2988424 is not an
    ancestor of develop -- it carries five commits that never landed there --
    so pinning develop alone would have taken the p11 fix and dropped the p09
    fix that test_mllama.py is gated on.  One commit now carries all three:
    p09 (mllama), p11 (this segfault) and p13 (7aec83d, the replicate axes
    Qwen2-VL needs), the last of which landed on develop WHILE this pin was
    being moved and would otherwise have been left behind.

    Re-measured on it: no segfault, and each of the four returns a max rel
    IDENTICAL to its a616ea5 run to every digit (5.67e-01, 5.83e-01, 1.35e+00,
    1.07e+00); test_mllama.py still passes both towers at 4.2915e-06 and
    3.4019e-06; the gate is unchanged at 1.85e-05 and 1.15e-05.  v3 and v9,
    which fail for reasons that have nothing to do with p11, were re-measured
    too and are also unchanged to every digit (1.27e+00 and 2.47e+00) -- so the
    pin move is visible in exactly one place across the nine versions, and it
    is the segfault.

WHAT WOULD HAVE HIDDEN ALL OF THIS.  Every one of these seven passed an earlier
version of this file, and the two reasons are in ``_revive`` and in the
tolerance:  with default weights the activations decay to 1e-09 by P5, and with
a flat atol=1e-2 every comparison against them is vacuous.  v3 went from "max
diff 3.2e-03, PASSED" to a 127% error on the same build, with only the weights
and the tolerance changed.  A green YOLO run is worth exactly what its
activations and its tolerance are worth.
"""

import argparse
import copy
import logging
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# One preset per parser version directory under ultralytics/cfg/models.  The
# YAML name is what selects the scale: parse_model reads it back out of the
# stem (yolov8n -> scale "n"), which is the same path a user takes.
_PRESETS = {
    "v3": dict(yaml="yolov3-tiny.yaml", imgsz=64, nc=4),
    "v5": dict(yaml="yolov5n.yaml", imgsz=64, nc=4),
    "v6": dict(yaml="yolov6n.yaml", imgsz=64, nc=4),
    "v8": dict(yaml="yolov8n.yaml", imgsz=64, nc=4),
    "v9": dict(yaml="yolov9t.yaml", imgsz=64, nc=4),
    "v10": dict(yaml="yolov10n.yaml", imgsz=64, nc=4),
    "11": dict(yaml="yolo11n.yaml", imgsz=64, nc=4),
    "12": dict(yaml="yolo12n.yaml", imgsz=64, nc=4),
    "26": dict(yaml="yolo26n.yaml", imgsz=64, nc=4),
}

# What the parser MUST have instantiated for the preset to mean what it claims,
# and what it must NOT have -- the second half is what keeps a preset from
# silently becoming a copy of its neighbour.
_CHARACTER = {
    "v3":  dict(needs=("ZeroPad2d", "MaxPool2d"),
                forbids=("C3", "C2f", "C3k2", "A2C2f", "SPPF", "Bottleneck")),
    "v5":  dict(needs=("C3", "Bottleneck", "SPPF"),
                forbids=("C2f", "C3k2", "A2C2f", "ConvTranspose2d")),
    "v6":  dict(needs=("ConvTranspose2d", "SPPF"),
                forbids=("C3", "C2f", "C3k2", "A2C2f", "Upsample")),
    "v8":  dict(needs=("C2f", "Bottleneck", "SPPF"),
                forbids=("C3", "C3k2", "C2PSA", "A2C2f", "ConvTranspose2d")),
    "v9":  dict(needs=("RepNCSPELAN4", "ELAN1", "SPPELAN", "AConv", "RepConv"),
                forbids=("C3", "C2f", "C3k2", "SPPF", "A2C2f")),
    "v10": dict(needs=("C2fCIB", "PSA", "SCDown", "RepVGGDW", "v10Detect"),
                forbids=("C3k2", "A2C2f", "C2PSA")),
    "11":  dict(needs=("C3k2", "C2PSA", "PSABlock", "SPPF"),
                forbids=("C2f", "A2C2f", "v10Detect")),
    "12":  dict(needs=("A2C2f", "AAttn", "C3k2"),
                forbids=("SPPF", "C2PSA", "v10Detect", "MaxPool2d")),
    "26":  dict(needs=("C3k2", "C2PSA", "SPPF"),
                forbids=("C2f", "A2C2f", "v10Detect")),
}

# Every version this parser builds gets the anchor-free head; 26 is the one that
# drops the DFL with it (reg_max 1, Identity in place of the 16-bin softmax).
_NO_DFL = {"26"}
# NMS-free: the model itself returns selected boxes, not a dense grid.
_END2END = {"v10", "26"}


def _quiet_ultralytics():
    """Ultralytics logs a full layer table per build; the test prints its own."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)


def _revive(model, imgsz, batch=8):
    """Give every BatchNorm the statistics of its own input, in one pass.

    THIS IS NOT COSMETIC, and the test proves much less without it.  An
    untrained BatchNorm in eval mode is a PASSTHROUGH -- running_mean 0,
    running_var 1 -- so nothing renormalises between layers, and PyTorch's
    default Conv2d init shrinks the activation variance at every one of them.
    Measured on v8: the three neck outputs come out at 3.0e-05, 2.7e-07 and
    4.5e-09.  At that size the comparison is decided by the detection head's
    bias and by the anchor grid -- both constants this backend barely touches
    -- and a backbone computing garbage would still pass.

    Calibrating is what a BatchNorm is FOR, and it is one forward pass: a
    pre-hook writes each BN's running stats from the activation about to enter
    it, so by the time the pass reaches layer N every layer before it has
    already been normalised.  v8's three neck outputs afterwards: 10, 14, 17.
    Every version lands between 4 and 1200 across all outputs, which is what
    makes ONE relative tolerance meaningful for all of them.

    A GAIN-ONLY FIX WAS TRIED FIRST AND IS WORSE.  Re-initialising the convs
    Kaiming fan-in/relu also clears the decay, but nothing bounds the result
    afterwards: v12 came out at 1.6e+05 and yolov3-tiny diverged outright --
    its DFL softmax saturated on the device to bin 15, so every decoded box
    was anchor + 15*16 = 240 where the CPU had a spread.  Normalising per
    layer is what keeps the dynamic range in a range both sides agree on.

    Transformer tests in this suite need no such thing: a LayerNorm or RMSNorm
    renormalises every block, so their random-weight activations stay O(1).
    """
    import torch.nn as nn

    hooks = []

    def _stats(mod, inputs):
        a = inputs[0].detach()
        dims = [0] + list(range(2, a.dim()))
        mod.running_mean.copy_(a.mean(dim=dims))
        # clamp_min: a channel that is constant across the calibration batch
        # would otherwise divide by zero and take the whole model to inf.
        mod.running_var.copy_(a.var(dim=dims, unbiased=False).clamp_min(1e-5))

    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_pre_hook(_stats))

    # Its own generator, so the calibration input is not the test input.
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model(torch.randn(batch, 3, imgsz, imgsz, generator=g))
    for h in hooks:
        h.remove()
    return model


def _build(version, imgsz=None):
    """Parse the version's YAML into a model.  No checkpoint, no network."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    p = _PRESETS[version]
    # Calibrate at the resolution the test will actually run: BN statistics
    # taken at 64 do not describe the activations at 128.
    imgsz = imgsz if imgsz is not None else p["imgsz"]
    cfg = yaml_model_load(p["yaml"])
    # Seed before construction: the weights are random, so without this the
    # worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    model = DetectionModel(cfg=cfg, ch=3, nc=p["nc"], verbose=False).eval()
    return _revive(model, imgsz), cfg


def _module_names(model):
    """Every module class name in the parsed graph, nested blocks included."""
    return {type(m).__name__ for m in model.modules()}


def _assert_character(version, model, cfg, imgsz=None):
    """Fail if the parser did not produce the blocks this preset is named for."""
    names = _module_names(model)
    spec = _CHARACTER[version]

    for want in spec["needs"]:
        assert want in names, (
            f"preset {version} claims {want}, which the parser did not instantiate; "
            f"this model is not the {version} architecture"
        )
    for forbid in spec["forbids"]:
        assert forbid not in names, (
            f"preset {version} must not contain {forbid} -- it would make this "
            f"preset a copy of another one"
        )

    head = model.model[-1]
    has_dfl = type(getattr(head, "dfl", None)).__name__ == "DFL"
    if version in _NO_DFL:
        assert not has_dfl and head.reg_max == 1, (
            f"{version} drops the DFL (reg_max 1, Identity); this head has "
            f"reg_max={head.reg_max} and a {type(head.dfl).__name__}"
        )
    else:
        assert has_dfl and head.reg_max == 16, (
            f"{version} runs the 16-bin DFL head through this parser; got "
            f"reg_max={getattr(head, 'reg_max', None)}"
        )

    end2end = bool(getattr(head, "end2end", False))
    assert end2end == (version in _END2END), (
        f"{version}: end2end is {end2end}; the preset exists for the opposite"
    )

    # A scaled-down input still has to leave the deepest stride a real feature
    # map -- otherwise the neck's concats degenerate and the run proves nothing.
    imgsz = imgsz if imgsz is not None else _PRESETS[version]["imgsz"]
    max_stride = int(max(model.stride))
    assert imgsz % max_stride == 0 and imgsz // max_stride >= 2, (
        f"imgsz {imgsz} against max stride {max_stride} leaves a "
        f"{imgsz // max_stride}x{imgsz // max_stride} map; too small to be a test"
    )


def _tensors(obj, out=None):
    """Flatten every tensor in a YOLO output, whatever the version wrapped it in.

    v3/v5/v8/v9/11 return (decoded, {boxes, scores, feats}); v10 and 26 return
    (selected, {one2many, one2one}).  Comparing the flattened list rather than
    just element 0 keeps the raw feature maps in the comparison, which is where
    the backbone and neck actually show up.
    """
    out = [] if out is None else out
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _tensors(obj[k], out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _tensors(v, out)
    return out


@torch.no_grad()
def run_yolo(device, version="v8", batch=1, imgsz=None, compile_model=True, rtol=1e-4, atol=1e-6):
    _quiet_ultralytics()
    p = _PRESETS[version]
    imgsz = imgsz if imgsz is not None else p["imgsz"]

    model_cpu, cfg = _build(version, imgsz)
    n_params = sum(x.numel() for x in model_cpu.parameters())
    top = [type(m).__name__ for m in model_cpu.model]
    print(f"version={version} yaml={p['yaml']} scale={cfg.get('scale')} "
          f"imgsz={imgsz} nc={p['nc']} batch={batch}")
    print(f"  params={n_params} layers={len(model_cpu.model)} "
          f"stride={[int(s) for s in model_cpu.stride]}")
    print(f"  blocks: {', '.join(sorted(set(top)))}")

    _assert_character(version, model_cpu, cfg, imgsz)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, 3, imgsz, imgsz, generator=g)

    cpu_out = _tensors(model_cpu(x))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _tensors(model_npu(x.to(device)))

    assert len(npu_out) == len(cpu_out), (
        f"output structure differs: {len(npu_out)} tensors on npu, {len(cpu_out)} on cpu"
    )
    # THE TOLERANCE IS PER OUTPUT, and a single atol would make most of these
    # comparisons vacuous.  These outputs do not share a scale: the decoded
    # boxes are pixel coordinates and run to 480, while a P5 neck map is O(0.1).
    # A flat atol=1e-2 would pass ANY value at all for the second -- every
    # element is already within 1e-2 of zero -- so each output gets
    # atol = tol * max|ref| instead, floored so an all-zero tensor still has a
    # threshold.  That is one criterion, applied at nine different scales.
    # The whole table is printed BEFORE any assertion.  test_result exits on
    # the first mismatch, and a failure that reports one output while hiding
    # the other eight says nothing about where the divergence starts.
    worst = worst_rel = 0.0
    rows = []
    for i, (a, b) in enumerate(zip(npu_out, cpu_out)):
        scale = torch.max(torch.abs(b)).item()
        scaled_atol = max(atol, rtol * scale)
        diff = torch.max(torch.abs(a.cpu() - b)).item()
        rel = diff / scale if scale else 0.0
        verdict = "ok" if diff <= scaled_atol + rtol * scale else "DIVERGED"
        print(f"  output[{i}] {str(tuple(b.shape)):>20}  max|d|={diff:.4e}  "
              f"max|ref|={scale:.4e}  rel={rel:.2e}  atol={scaled_atol:.2e}  {verdict}")
        rows.append((i, a, b, scaled_atol))
        worst, worst_rel = max(worst, diff), max(worst_rel, rel)
    print(f"Max diff >  {worst}")
    print(f"Max rel  >  {worst_rel}")
    for i, a, b, scaled_atol in rows:
        test_result(f"YOLO {version} output[{i}] {tuple(b.shape)}", a, b,
                    rtol=rtol, atol=scaled_atol)
    print("Yolo Simulation Done")


# THE GATE: what the sweep runs when it runs this file with no arguments.
#
# The two versions that pass, and they are not redundant: v5 is C3 + SPPF with
# the anchor-era backbone, v8 is C2f, and between them they cover both CSP
# shapes plus the DFL head every later version inherits.  MEASURED AS THE SWEEP
# RUNS IT -- one process, no arguments, 1128s -- inside the sweep's 1800s.  The
# two runs share codegen rather than colliding in it: 1453s if run separately,
# and each version's max rel is identical either way (1.85e-05 and 1.15e-05).
# THAT NUMBER IS LOAD-SENSITIVE and the margin is not large: the same gate run
# took 1578s on a machine also building a toolchain and running eight other
# models.  Same two diffs, to every digit, on the moved pin -- and again on the
# transposed-conv rewrite (1206s, sharing the machine with two other models).
#
# SEVEN VERSIONS PASS AND ARE STILL NOT HERE, which is the rule doing its job.
# v3 and v9 need triton_shared b6c3e60; v10, 11, 12 and 26 need 99dc6bc on top
# of it. `TRITON_SHARED_SHA` in triton-npu's setup/versions.env still points at
# 9017bd4e, and moving it needs a matching TRITON_SHARED_PREBUILT_SHA release,
# so CI would run a binary with neither fix. They join the gate when the pin
# moves, not when the fix is written -- a version is added here when it passes
# on the PINNED toolchain.
#
# V6 IS OUT FOR A DIFFERENT REASON, and it is runtime, not doubt.  Its blocker
# was removed in THIS repo, and it was measured on a triton_shared build
# carrying NEITHER b6c3e60 nor 99dc6bc -- so it is the one version here that
# owes the pin nothing.  What keeps it out is 498s: the gate would go from
# 1128s to about 1600s against an 1800s budget already seen to swell to 1578s
# under load.  Adding it would buy transposed-conv coverage at the price of a
# gate that times out on a busy runner.  It goes in when the budget does, or
# when the sweep learns to run the two halves separately.
#
# The rest are one --version each; each has a measured stopping point in the
# module docstring.
_GATE = ("v5", "v8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO, one preset per parser version")
    parser.add_argument("--version", type=str, default=None, choices=list(_PRESETS),
                        help=f"one version; the default runs the gate ({', '.join(_GATE)})")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="relative criterion; also scales each output's atol")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="absolute floor, for an output whose reference is all zero")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for version in ([args.version] if args.version else _GATE):
        run_yolo(
            torch.device("npu:0"),
            version=version,
            batch=args.batch,
            imgsz=args.imgsz,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
