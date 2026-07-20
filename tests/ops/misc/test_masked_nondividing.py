"""Masked-DMA non-dividing regression tests.

Dropping the tile-divisibility constraint (is_dim_dividable / must_divide_dim and the
_index_expr / convert_indirect_indexing RecompileSignal) means a non-dividing shape no
longer falls back to a dividing tile -- correctness now rests entirely on the masked-DMA
[low, high) clamp + reduction-identity fill. These cases pin that path: what used to be a
(slow but safe) recompile is now a silent wrong answer if the clamp regresses.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def test_pad_2d(device, size=(5, 10), pad=(2, 3, 1, 1)):
    # 2-D constant pad on a non-dividing shape -> exercises the per-dim greedy pad recovery
    # in _masked_bounds (both padded axes must be attributed to the right tile dim).
    def fn(x):
        return F.pad(x, pad)
    x = torch.randn(size).to(device=device)
    res = torch.compile(dynamic=False)(fn)(x)
    test_result(f"Pad2D {size} {pad}", res, fn(x.cpu()))


def test_sum_over_broadcast(device, size=(5, 13)):
    # sum over a non-dividing dim with a broadcast operand -- the tail fill must be the sum
    # identity 0, NOT -inf (regression guard for the exp-origin substring bug: an origin
    # name containing "exp" like expand/expm1 must not trigger the log-sum-exp -inf fill).
    def fn(a, b):
        return (a * b).sum(dim=1)
    a = torch.randn(size).to(device=device)
    b = torch.randn(size[0], 1).to(device=device)
    res = torch.compile(dynamic=False)(fn)(a, b)
    test_result(f"SumOverBroadcast {size}", res, fn(a.cpu(), b.cpu()))


def test_softmax_nondividing(device, size=(4, 10)):
    # genuine log-sum-exp reduction: the exp path must still fill -inf and stay correct.
    def fn(a):
        return F.softmax(a, dim=1)
    a = torch.randn(size).to(device=device)
    res = torch.compile(dynamic=False)(fn)(a)
    test_result(f"Softmax {size}", res, fn(a.cpu()))


def test_elementwise_nondividing(device, size=(7, 13)):
    def fn(a, b):
        return torch.relu(a) + b
    a = torch.randn(size).to(device=device)
    b = torch.randn(size).to(device=device)
    res = torch.compile(dynamic=False)(fn)(a, b)
    test_result(f"Elementwise {size}", res, fn(a.cpu(), b.cpu()))


def test_gather_nondividing(device, rows=13, cols=10, n=17):
    # indirect (gather) on a non-dividing shape -- the divisibility RecompileSignal is gone.
    def fn(x, idx):
        return x[idx]
    x = torch.randn(rows, cols).to(device=device)
    idx = torch.randint(0, rows, [n]).to(device=device)
    res = torch.compile(dynamic=False)(fn)(x, idx)
    test_result(f"Gather [{rows},{cols}]->{n}", res, fn(x.cpu(), idx.cpu()))


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pad_2d(device, (5, 10), (2, 3, 1, 1))
    test_pad_2d(device, (7, 13), (1, 1, 3, 2))
    test_pad_2d(device, (1, 3, 10, 10), (2, 1, 1, 2))
    test_sum_over_broadcast(device, (5, 13))
    test_softmax_nondividing(device, (4, 10))
    test_elementwise_nondividing(device, (7, 13))
    test_gather_nondividing(device)
