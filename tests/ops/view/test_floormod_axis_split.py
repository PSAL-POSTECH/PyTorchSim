"""Floor/mod index handling: axis-split (aligned) + graph-copy (incompatible).

Covers the index-expression shapes that view/reshape/tile/group ops produce and
how the frontend handles them:

  - aligned floor/mod (single iter var, divisor divides extent): removed by
    axis-split at the scheduling layer. group_norm, repeat, repeat_interleave,
    permute+reshape (mixed-radix).
  - incompatible radices on a shared axis (case 5, e.g. a[c//2] + b[c%3]): the
    conflicting operand is realized by graph-copy so the consumer reads it affine
    and the remainder is axis-split's.
  - cross-axis / multi-variable floor/mod argument (case 7, e.g. (3*p0+p1)//4 from
    a transpose+reshape feeding a broadcast/softmax/layernorm that keeps the dims
    separate): graph-copy materializes the multi-var operand with copy_input (which
    forces a copy of a view, unlike realize()); the copy kernel iterates the
    operand's own shape so its index collapses to single-var for axis-split.

Both features are always on; graph-copy installs its lowering hook at import.

Not in the CI allowlist (pytorchsim_test.yml) -- local feature/regression test.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from PyTorchSimFrontend.mlir import graph_copy
graph_copy.install()


def _run(device, name, fn, *inputs):
    torch.manual_seed(0)
    opt = torch.compile(dynamic=False)(fn)
    res = opt(*[t.to(device=device) for t in inputs])
    ref = fn(*[t.cpu() for t in inputs])
    test_result(name, res, ref, rtol=1e-3, atol=1e-3)


# --- aligned floor/mod: handled by axis-split ---------------------------------
def test_group_norm(device):
    _run(device, "group_norm c//(C/G)", lambda x: F.group_norm(x, 3), torch.randn(2, 6, 4, 4))


def test_repeat(device):
    # tile -> ModularIndexing(c, 1, n)
    _run(device, "repeat (mod)", lambda x: x.repeat(1, 2) + 1.0, torch.randn(4, 8))


def test_repeat_interleave(device):
    # -> FloorDiv(c, k)
    _run(device, "repeat_interleave (floor)",
         lambda x: torch.repeat_interleave(x, 2, dim=1) + 1.0, torch.randn(2, 4, 8))


def test_permute_reshape(device):
    # permute+reshape -> single-var mixed-radix floor/mod
    _run(device, "permute+reshape (mixed-radix)",
         lambda x: x.permute(0, 2, 1).reshape(2, 12) + 1.0, torch.randn(2, 3, 4))


def test_three_level_mixed_radix(device):
    # reshape+permute+reshape -> chain [1,4,12,24]; the 3-level split leaves a
    # residual FloorDiv that simplify_with_ranges cannot fold -> _fold_with_ranges.
    _run(device, "3-level mixed-radix",
         lambda x: x.reshape(2, 3, 2, 4).permute(0, 2, 1, 3).reshape(2, 24) + 1.0,
         torch.randn(2, 6, 4))


def test_pixel_shuffle(device):
    # splits two spatial axes -> 5D logical tile; the decompose-transfer pass peels
    # the outer dims into an affine.for nest with the lane-banked physical SRAM offset.
    _run(device, "pixel_shuffle (>4D peel)",
         lambda x: F.pixel_shuffle(x, 2) + 1.0, torch.randn(1, 8, 4, 4))


# --- incompatible radices (case 5): handled by graph-copy ---------------------
def test_incompatible_radix(device):
    # a[c//2] + b[c%3] on axis c=6 : floor-by-2 vs mod-by-3 (not a chain)
    _run(device, "incompat a[c//2]+b[c%3]",
         lambda a, b: torch.repeat_interleave(a, 2, dim=1) + b.repeat(1, 2),
         torch.randn(2, 3), torch.randn(2, 3))


# --- cross-axis multi-var floor/mod (case 7): handled by graph-copy copy_input -
def test_case7_reshape_broadcast(device):
    # (3*p0+p1)//4 from transpose+reshape feeding an elementwise broadcast consumer
    _run(device, "case7 reshape+broadcast",
         lambda x, y: x.t().reshape(8, 3) + y, torch.randn(4, 6), torch.randn(8, 1))


def test_case7_softmax_reshape(device):
    # same multi-var floor feeding a reduction (softmax over the kept-separate dim)
    _run(device, "case7 softmax(reshape)",
         lambda x: F.softmax(x.t().reshape(8, 3), dim=1), torch.randn(4, 6))


def test_case7_layernorm_reshape(device):
    _run(device, "case7 layernorm(reshape)",
         lambda x: F.layer_norm(x.t().reshape(8, 3), (3,)), torch.randn(4, 6))


if __name__ == "__main__":
    device = torch.device("npu:0")
    with torch.no_grad():
        test_group_norm(device)
        test_repeat(device)
        test_repeat_interleave(device)
        test_permute_reshape(device)
        test_three_level_mixed_radix(device)
        test_pixel_shuffle(device)
        test_incompatible_radix(device)
        test_case7_reshape_broadcast(device)
        test_case7_softmax_reshape(device)
        test_case7_layernorm_reshape(device)
