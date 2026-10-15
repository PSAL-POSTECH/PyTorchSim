"""Reduction blocks on a machine of 128 lanes and 1 MiB of scratchpad per lane: a reduction whose
largest read runs along the kept axis takes that axis as wide as its unpadded power-of-two part
allows; any other reduction keeps today's blocks."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from pytorchsim.codegen import tiling

LANES, LANE_BYTES = 128, 1 << 20


def _blocks(xnumel, rnumel, elem_bytes=2):
    return tiling.kept_axis_contiguous_blocks(xnumel, rnumel, elem_bytes, LANES, LANE_BYTES,
                                              tiling.triton_max_numel())


def test_gemv_blocks_follow_the_contiguous_kept_axis():
    """1 x K times K x N: N is the kept, contiguous axis; XBLOCK is its widest unpadded power of two."""
    want = {(4096, 4096): (4096, 512), (8192, 8192): (8192, 256), (2048, 2048): (2048, 1024),
            (3072, 8192): (1024, 2048), (5120, 8192): (1024, 2048), (768, 768): (256, 1024),
            (384, 384): (128, 512), (1000, 4096): (1024, 2048)}
    got = {k: _blocks(*k) for k in want}
    test_result("GemvBlocks", torch.tensor(list(got.values())), torch.tensor(list(want.values())))


def test_lane_cover_and_budget_are_never_crossed():
    """XBLOCK is never under the lane cover and XBLOCK * R0_BLOCK never over the tile budget."""
    per_lane = tiling.reduction_block_for(1 << 62, 2, LANE_BYTES)
    bad = []
    for x in (1, 64, 128, 384, 1000, 3072, 4096, 11008, 65536):
        for r in (8, 512, 4096, 50257):
            xb, rb = _blocks(x, r)
            if xb < min(LANES, 1 << (x - 1).bit_length()) or xb * rb > LANES * per_lane:
                bad.append((x, r, xb, rb))
    test_result("BlocksWithinBudget", torch.tensor(len(bad)), torch.tensor(0))


def test_only_an_outer_hint_reads_as_kept_axis_contiguous():
    """Inductor's OUTER hint (no read runs along the reduced axis) is the only one that says x."""
    from torch._inductor.runtime.hints import ReductionHint
    from pytorchsim.frontend.inductor_patches import _kept_axis_contiguous
    feats = lambda h, red=True: type("F", (), {"is_reduction": lambda self: red,
                                               "get_reduction_hint": lambda self: h})()
    got = [_kept_axis_contiguous(feats(h)) for h in (ReductionHint.OUTER, ReductionHint.INNER,
                                                     ReductionHint.DEFAULT)]
    got.append(_kept_axis_contiguous(feats(ReductionHint.OUTER, red=False)))
    test_result("OuterHintOnly", torch.tensor([g == "x" for g in got]),
                torch.tensor([True, False, False, False]))


def test_other_reductions_keep_todays_blocks():
    """contiguous None or 'r' (a row reduction) gives exactly the blocks of before."""
    kernel = type("K", (), {"inside_reduction": True})()
    numels = {"xnumel": 4096, "r0_numel": 1024}
    args = [{"dtype": "float32"}]
    base = tiling.fixed_config_for(kernel, numels, args)
    same = [tiling.fixed_config_for(kernel, numels, args, contiguous=c) == base for c in (None, "r")]
    test_result("RowReductionUnchanged", torch.tensor(same), torch.tensor([True, True]))


if __name__ == "__main__":
    test_gemv_blocks_follow_the_contiguous_kept_axis()
    test_lane_cover_and_budget_are_never_crossed()
    test_only_an_outer_hint_reads_as_kept_axis_contiguous()
    test_other_reductions_keep_todays_blocks()
