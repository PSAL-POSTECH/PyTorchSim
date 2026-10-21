"""Reduction blocks on a machine of 128 lanes and 1 MiB of scratchpad per lane: a reduction whose
largest read runs along the kept axis takes that axis as wide as its unpadded power-of-two part
allows; a reduction whose rows fill fewer lanes than the machine has takes the reduced axis on the
lanes and budgets it over all of them; any other reduction takes rows of its R0_BLOCK up to the budget."""
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


def test_kept_axis_is_the_unit_stride_axis_of_the_largest_read():
    """A GEMV reads its weight unit-stride along the kept axis at any width (24576 rows is past
    Inductor's num_sm * 64, where its hint stays DEFAULT); a row reduction and a pointwise kernel do not."""
    import sympy
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.sizevars import SizeVarAllocator
    from torch._inductor.virtualized import V
    from pytorchsim.frontend.inductor_patches import _kept_axis_contiguous
    c0, c1 = sympy.symbols("c0 c1", integer=True, nonnegative=True)
    graph = type("G", (), {"sizevars": SizeVarAllocator(), "get_dtype": lambda self, n: torch.bfloat16})()

    def feats(n, reads, red=True):
        rw = type("RW", (), {"reads": reads})()
        node = type("N", (), {"_sizes": ((n,), (4096,)), "read_writes": rw})()
        return type("F", (), {"is_reduction": lambda self: red, "reduction_nodes": lambda self: [node]})()

    gemv = lambda n: [MemoryDep("x", c1, (c0, c1), (n, 4096)), MemoryDep("w", c0 + n * c1, (c0, c1), (n, 4096))]
    rows = [MemoryDep("a", 4096 * c0 + c1, (c0, c1), (8192, 4096)), MemoryDep("g", c1, (c0, c1), (8192, 4096))]
    with V.set_graph_handler(graph):
        got = [_kept_axis_contiguous(feats(4096, gemv(4096))), _kept_axis_contiguous(feats(24576, gemv(24576))),
               _kept_axis_contiguous(feats(8192, rows)), _kept_axis_contiguous(feats(4096, gemv(4096), red=False))]
    test_result("KeptAxisFromStrides", torch.tensor([g == "x" for g in got]),
                torch.tensor([True, True, False, False]))


def test_row_reduction_rows_fill_the_tile_budget():
    """A reduction along its contiguous axis takes as many rows of R0_BLOCK as the tile budget holds:
    q_norm's 128-wide rows go 128 -> 16384 a tile, a 4096-wide row 128 -> 512, a few rows stay covered."""
    kernel = type("K", (), {"inside_reduction": True})()
    args = [{"dtype": "bfloat16"}]
    cases = {(32768, 128): (16384, 128), (8192, 128): (8192, 128), (1024, 4096): (512, 4096),
             (128, 4096): (128, 4096), (1, 4096): (1, 4096)}
    got = {k: tuple(tiling.fixed_config_for(kernel, {"xnumel": k[0], "r0_numel": k[1]}, args)[b]
                    for b in ("XBLOCK", "R0_BLOCK")) for k in cases}
    test_result("RowReductionRows", torch.tensor(list(got.values())), torch.tensor(list(cases.values())))


def test_other_reductions_keep_todays_blocks():
    """contiguous None or 'r' (a row reduction) gives the same blocks."""
    kernel = type("K", (), {"inside_reduction": True})()
    numels = {"xnumel": 4096, "r0_numel": 1024}
    args = [{"dtype": "float32"}]
    base = tiling.fixed_config_for(kernel, numels, args)
    same = [tiling.fixed_config_for(kernel, numels, args, contiguous=c) == base for c in (None, "r")]
    test_result("RowReductionUnchanged", torch.tensor(same), torch.tensor([True, True]))


def test_few_rows_budget_the_reduced_axis_over_every_lane():
    """A full reduction (XBLOCK 1) and a few rows' reduction take R0_BLOCK up to every lane's budget
    over the rows, covering the extent; rows that fill the lanes keep today's block."""
    kernel = type("K", (), {"inside_reduction": True})()
    args = [{"dtype": "float32"}]
    per_lane = tiling.reduction_block_for(1 << 62, 4, LANE_BYTES)
    whole = min(LANES * per_lane, tiling.triton_max_numel())
    cases = [({"xnumel": 1, "r0_numel": 4096 * 4096}, whole),
             ({"xnumel": 1, "r0_numel": 96 * 128}, 16384),
             ({"xnumel": 8, "r0_numel": 16384}, 16384),
             ({"xnumel": 8, "r0_numel": 1 << 24}, whole // 8),
             ({"xnumel": 4096, "r0_numel": 1024},
              tiling.fixed_config_for(kernel, {"xnumel": 4096, "r0_numel": 1024}, args)["R0_BLOCK"])]
    got = [tiling.fixed_config_for(kernel, n, args)["R0_BLOCK"] for n, _ in cases]
    test_result("FewRowsLaneBlocks", torch.tensor(got), torch.tensor([w for _, w in cases]))


if __name__ == "__main__":
    test_gemv_blocks_follow_the_contiguous_kept_axis()
    test_lane_cover_and_budget_are_never_crossed()
    test_kept_axis_is_the_unit_stride_axis_of_the_largest_read()
    test_row_reduction_rows_fill_the_tile_budget()
    test_other_reductions_keep_todays_blocks()
    test_few_rows_budget_the_reduced_axis_over_every_lane()
