"""An in-place op that writes SOME of a buffer must leave the rest alone.

`scatter_`, `index_put_` and `index_copy_` write the positions an index names
and nothing else. Every other element has to survive from the tensor they were
called on. On this backend they do not: the untouched elements come back 0.

    fill 7.0 then scatter_    cpu [7.0, 1.54, 7.0, 7.0, ...]
                              npu [0.0, 1.54, 0.0, 0.0, ...]

WHY IT IS NOT A ZERO TEST, and this is the trap the file exists to hold open.
Filling with 0.0 makes the SAME defect pass -- "the base was copied and it was
zero" and "the base was never copied" are the same sixteen zeros. The fill is
7.0 for that reason and must stay non-zero; a future simplification to
`torch.zeros` retires the test without changing a line of its name.

WHERE IT COMES FROM. Inductor plans this as two kernels and its plan is right:

    ..._full_like_scatter_0(out_ptr0=buf0, xnumel=16)     fills all sixteen
    ..._full_like_scatter_1(..., out_ptr0=buf0, xnumel=4) writes four
        inductor_meta: mutated_arg_names: ['out_ptr0']

The second kernel MUTATES buf0. `mutated_arg_names` is where Inductor says so,
and `PyTorchSimCodegen/kernel_spec.py` deliberately does not read it -- it decides
each argument's role from the stores in the source instead, because the table
overstates on SD1.5 (two kernels list an `in_out_ptr0` they never write, and
believing it produced eight divergence reports about values nothing
materialised).

That reasoning is sound in the direction it was measured and blind in the other.
A store tells you the argument WAS written; it cannot tell you whether all of it
was. `out_ptr0` here is stored to, so it is classified `out`, so the runtime
hands the kernel a fresh buffer rather than the one holding the fill -- and the
twelve positions the scatter never touches keep the fresh buffer's zeros.

`masked_fill_` is here as the boundary rather than for coverage: it is
`where(mask, value, self)`, so it writes EVERY element and reads `self` to do
it. It passes, and it is the reason the defect is about partial writes and not
about in-place ops.

MEASURED 2026-08-14 on transformers 5.15.0, tnpu 983eee4, before any fix:

    scatter_      npu has 12 zeros of 16, the untouched positions exactly
    index_put_    12 of 16
    index_copy_   12 of 16
    masked_fill_  0 of 16, passes

FOUND IN Llama 4, whose MoE router fills with -inf and scatters the top-k
values in, so sigmoid turns the unselected experts into 0. With the base lost,
every expert arrives at 0.5 instead and top-1 routing becomes a blend of all
four. That model reads as "close but wrong", which is what a silently dropped
base looks like from the far end.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

T, E = 4, 4
#: NOT ZERO ON PURPOSE -- see the module docstring. Zero makes the defect pass.
FILL = 7.0


def _base(vals):
    return torch.full_like(vals[:, :1].expand(T, E).contiguous(), FILL)


def test_scatter_keeps_the_base(device):
    def f(vals, idx):
        return _base(vals).scatter_(1, idx, vals)

    g = torch.Generator().manual_seed(0)
    vals = torch.randn(T, 1, generator=g)
    idx = torch.randint(0, E, (T, 1), generator=g)

    cpu_out = f(vals, idx)
    npu_out = torch.compile(dynamic=False)(f)(vals.to(device), idx.to(device))
    test_result("scatter_ keeps the untouched elements", npu_out, cpu_out)


def test_index_put_keeps_the_base(device):
    def f(vals, idx):
        rows = torch.arange(T, device=vals.device)
        return _base(vals).index_put_((rows, idx.squeeze(1)), vals.squeeze(1))

    g = torch.Generator().manual_seed(1)
    vals = torch.randn(T, 1, generator=g)
    idx = torch.randint(0, E, (T, 1), generator=g)

    cpu_out = f(vals, idx)
    npu_out = torch.compile(dynamic=False)(f)(vals.to(device), idx.to(device))
    test_result("index_put_ keeps the untouched elements", npu_out, cpu_out)


def test_index_copy_keeps_the_base(device):
    def f(vals):
        col = torch.zeros(1, dtype=torch.long, device=vals.device)
        return _base(vals).index_copy_(1, col, vals)

    g = torch.Generator().manual_seed(2)
    vals = torch.randn(T, 1, generator=g)

    cpu_out = f(vals)
    npu_out = torch.compile(dynamic=False)(f)(vals.to(device))
    test_result("index_copy_ keeps the untouched elements", npu_out, cpu_out)


def test_masked_fill_is_the_boundary(device):
    """Writes every element, so it cannot lose a base. Passes today.

    Kept so a fix aimed at partial writes is not written in a way that only
    covers indexed ones: this is the shape that already worked.
    """
    def f(vals, idx):
        mask = torch.arange(E, device=vals.device)[None, :] == idx
        return _base(vals).masked_fill_(mask, 99.0)

    g = torch.Generator().manual_seed(3)
    vals = torch.randn(T, 1, generator=g)
    idx = torch.randint(0, E, (T, 1), generator=g)

    cpu_out = f(vals, idx)
    npu_out = torch.compile(dynamic=False)(f)(vals.to(device), idx.to(device))
    test_result("masked_fill_ writes every element", npu_out, cpu_out)


if __name__ == "__main__":
    device = torch.device("npu:0")
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as tests/models/Llama.
    test_scatter_keeps_the_base(device)
    test_index_put_keeps_the_base(device)
    test_index_copy_keeps_the_base(device)
    test_masked_fill_is_the_boundary(device)
