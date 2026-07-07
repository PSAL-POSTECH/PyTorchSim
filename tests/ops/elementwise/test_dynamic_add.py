"""Dynamic-shape elementwise add and reduction on the C++ trace path.

A single torch.compile(dynamic=True) kernel compiles to one trace producer .so
and is simulated at several input sizes -- the producer reads its loop bound from
shape_args at runtime, so the same .so serves any size. This exercises the
dynamic-shape pipeline end to end (symbolic tiling -> symbolic MLIR loop bound ->
shape_args producer -> per-tile cost table -> runtime shape via the attribute
file, plus a shape-agnostic Spike validation binary for the output values).

Sizes deliberately include ones that do NOT divide the tile, and one smaller than
a whole tile: a symbolic extent cannot be proved dividing at compile time, so the
masked-DMA clamp (affine.min(tile, extent - base), with the runtime extent as an
affine symbol operand) is always emitted and trims the last partial tile on both
the loads and the store. The reduction case is what actually proves the clamp:
an elementwise add tolerates a garbage tail (it lands in dead output slots),
whereas a sum would fold the tail into the result.
"""
import os
import sys

import torch
import torch._dynamo

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

# 1024/2048 divide the tile; 1000 leaves a tail; 100 is smaller than one tile.
SIZES = (1024, 2048, 1000, 100)


def test_dynamic_add(device, sizes=SIZES):
    def add(a, b):
        return a + b

    # Compile once with a symbolic shape; run at every size from the same .so.
    opt_fn = torch.compile(dynamic=True)(add)
    for n in sizes:
        x = torch.randn(n).to(device=device)
        y = torch.randn(n).to(device=device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        res = opt_fn(x, y)
        out = add(x.cpu(), y.cpu())
        test_result(f"DynamicAdd(N={n})", res, out)


def test_dynamic_sum(device, sizes=SIZES):
    def total(a):
        return a.sum()

    # A reduction folds every loaded element into the result, so a non-dividing
    # size here fails unless the tail is really trimmed (masked_fill = the
    # reduction identity for the excluded lanes).
    opt_fn = torch.compile(dynamic=True)(total)
    for n in sizes:
        x = torch.randn(n).to(device=device)
        torch._dynamo.mark_dynamic(x, 0)
        res = opt_fn(x)
        out = total(x.cpu())
        test_result(f"DynamicSum(N={n})", res, out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_dynamic_add(device)
    test_dynamic_sum(device)
