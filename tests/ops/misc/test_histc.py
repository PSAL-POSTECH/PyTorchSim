import os
import sys
import torch
import torch._dynamo
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

from pytorchsim.frontend.inductor.post_grad_pass import pytorchsim_post_grad_pass

aten = torch.ops.aten


def _histc(x, bins, lo, hi):
    return torch.histc(x, bins=bins, min=lo, max=hi)


def test_histc_integral(device, n=64, bins=8):
    # An op with no npu kernel falls back, and the fallback is the CPU one, so
    # what this device supports is what ATen's CPU kernel supports. histc is a
    # place where that is narrower than CUDA:
    #
    #     NotImplementedError: "histogram_cpu" not implemented for 'Int'
    #
    # transformers' MoE router (integrations/moe.py) counts tokens per expert
    # with histc and picks its input dtype by DEVICE NAME -- float for "cpu"
    # and "mps", integral otherwise -- so `npu` gets the branch written for
    # CUDA and the fallback then lands on the very kernel the first branch
    # exists to avoid. That is what stopped DeepSeek-V3 under transformers
    # 5.15.0; post_grad_pass.rewrite_aten_histc counts it on the device.
    #
    # Both integral widths are here because they take different paths back:
    # the counts return in the INPUT's dtype, which is the contract the CUDA
    # branch promised its caller.
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.int32, torch.int64):
        cpu_x = torch.randint(0, bins, (n,), generator=g).to(dtype)
        cpu_out = torch.histc(cpu_x.float(), bins=bins, min=0,
                              max=bins - 1).to(dtype)

        opt_fn = torch.compile(dynamic=False)(_histc)
        npu_out = opt_fn(cpu_x.to(device=device), bins, 0, bins - 1)

        assert npu_out.dtype == dtype, (
            f"histc should return the input's dtype: {npu_out.dtype} != {dtype}")
        check_result(f"histc {dtype}", npu_out, cpu_out)


def test_histc_float_unchanged(device, n=64, bins=8):
    """A floating histc is counted on the device too, and comes back in its own dtype."""
    g = torch.Generator().manual_seed(1)
    cpu_x = torch.rand(n, generator=g) * (bins - 1)
    cpu_out = torch.histc(cpu_x, bins=bins, min=0, max=bins - 1)

    opt_fn = torch.compile(dynamic=False)(_histc)
    npu_out = opt_fn(cpu_x.to(device=device), bins, 0, bins - 1)

    assert npu_out.dtype == torch.float32, npu_out.dtype
    check_result("histc float32", npu_out, cpu_out)


def _unlowered(gm):
    """The ops of gm Inductor has no lowering for; the device pass runs after decomposition, so any it
    inserts must lower as they are."""
    from torch._inductor.lowering import lowerings
    return sorted({str(m.target) for m in gm.graph.nodes if callable(m.target)
                   and m.target not in lowerings and m.target.overloadpacket not in lowerings})


def _histc_graph(x, bins, lo, hi):
    """A graph of one aten.histc of x as the post-grad passes see it; built by hand, since make_fx cannot
    trace an integral histc on the CPU."""
    mode = FakeTensorMode()
    g = fx.Graph()
    p = g.placeholder("x")
    p.meta["val"] = mode.from_tensor(x)
    h = g.call_function(aten.histc.default, (p, bins, lo, hi))
    with mode:
        h.meta["val"] = torch.empty(bins, dtype=x.dtype)
    g.output(h)
    return fx.GraphModule(torch.nn.Module(), g)


def test_pass_counts_on_the_device(n=64):
    """The device pass alone leaves no aten.histc and counts what aten counts: bins that are the values,
    wider edges, int8 past 127 bins, floats with values on and past the edges, and min == max."""
    g = torch.Generator().manual_seed(2)
    ints = torch.randint(-4, 20, (n,), generator=g)
    floats = torch.cat([torch.rand(n, generator=g) * 3 - 1, torch.tensor([0., 0.5, 1., 1.0000001, float("nan")])])
    cases = (
        ("values", ints.clamp(0, 7).to(torch.int32), 8, 0, 7),
        ("wide edges", ints.to(torch.int32), 8, 0, 15),
        ("int8 past 127 bins", torch.tensor([0, 1, 127, -100, -100, -100], dtype=torch.int8), 200, 0, 199),
        ("float", floats, 4, 0, 1),
        ("float min == max", floats.nan_to_num(0.0), 5, 0, 0),
        ("int64 min == max", ints, 6, 0, 0),
        ("one value", torch.full((7,), 3, dtype=torch.int64), 4, 0, 0),
    )
    for name, x, bins, lo, hi in cases:
        gm = _histc_graph(x, bins, lo, hi)
        pytorchsim_post_grad_pass(gm)
        left = [m for m in gm.graph.nodes if m.target is aten.histc.default]
        assert not left, f"{name}: {len(left)} aten.histc left"
        assert not _unlowered(gm), f"{name}: no Inductor lowering for {_unlowered(gm)}"
        ref = torch.histc(x.double() if not x.is_floating_point() else x, bins, lo, hi).to(x.dtype)
        out = gm(x)
        assert torch.equal(out, ref), f"{name}: {out.tolist()} != {ref.tolist()}"
        print(f"histc pass {name}: {out.tolist()}")
    for name, x, lo, hi in (("int edges that truncate to one", ints.to(torch.int32), 0.2, 0.8),
                            ("float max below min", floats, 1.0, 0.0)):
        try:
            pytorchsim_post_grad_pass(_histc_graph(x, 4, lo, hi))
        except RuntimeError as e:
            assert "max must be larger than min" in str(e), f"{name}: {e}"
        else:
            raise AssertionError(f"{name}: [{lo}, {hi}] was not refused")


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_pass_counts_on_the_device()
    test_histc_integral(device)
    test_histc_float_unchanged(device)
