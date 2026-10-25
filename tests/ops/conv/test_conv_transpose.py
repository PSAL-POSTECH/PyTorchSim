"""A transposed convolution on the npu runs as its direct equivalent, against the CPU's conv_transpose.
Cases: stride_pad, groups_dilated, one_d, batchless, one_d_no_bias, and pass_graph (the post-grad pass leaves no
transposed convolution); run with [case ...], all when none is named."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")

#: name -> (input shape, weight shape, stride, padding, output_padding, groups, dilation)
CASES = {
    "stride_pad": ((1, 8, 9, 9), (8, 16, 3, 3), 2, 1, 1, 1, 1),
    "groups_dilated": ((2, 8, 7, 7), (8, 4, 3, 3), 1, 0, 0, 2, 2),
    "one_d": ((2, 8, 16), (8, 16, 4), 2, 1, 0, 1, 1),
    "batchless": ((8, 9, 9), (8, 16, 3, 3), 2, 1, 0, 1, 1),
}


def _conv_transpose(x, w, b, stride, padding, output_padding, groups, dilation):
    fn = F.conv_transpose1d if w.dim() == 3 else F.conv_transpose2d
    return fn(x, w, b, stride=stride, padding=padding, output_padding=output_padding,
              groups=groups, dilation=dilation)


def _inputs(name):
    xs, ws, *_ = CASES[name]
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(*xs, generator=g), torch.randn(*ws, generator=g)
    groups = CASES[name][5]
    return x, w, torch.randn(ws[1] * groups, generator=g)


def run_case(name):
    x, w, b = _inputs(name)
    params = CASES[name][2:]
    ref = _conv_transpose(x, w, b, *params)
    torch._dynamo.reset()
    f = torch.compile(lambda x, w, b: _conv_transpose(x, w, b, *params), dynamic=False)
    with torch.no_grad():
        out = f(x.to(device), w.to(device), b.to(device))
    check_result(f"ConvTranspose {name}", out, ref, rtol=1e-3, atol=1e-3)
    return True


def run_no_bias(name):
    """A 1-D transposed conv with no bias: its direct replacement is traced with (x, weight) alone."""
    x, w, _ = _inputs("one_d")
    params = CASES["one_d"][2:]
    ref = _conv_transpose(x, w, None, *params)
    torch._dynamo.reset()
    f = torch.compile(lambda x, w: _conv_transpose(x, w, None, *params), dynamic=False)
    with torch.no_grad():
        out = f(x.to(device), w.to(device))
    check_result("ConvTranspose one_d_no_bias", out, ref, rtol=1e-3, atol=1e-3)
    return True


def run_pass_graph(name):
    """On a traced graph: the pass rewrites every transposed convolution into a direct one."""
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    from pytorchsim.frontend.inductor import postgrad as pg
    x, w, b = _inputs("stride_pad")
    params = CASES["stride_pad"][2:]
    gm = make_fx(lambda x, w, b: _conv_transpose(x, w, b, *params),
                 decomposition_table=select_decomp_table(), tracing_mode="fake")(x, w, b)
    pg.CALLS.clear()
    pg.pytorchsim_post_grad_pass(gm)
    convs = gm.graph.find_nodes(op="call_function", target=torch.ops.aten.convolution.default)
    left = [c for c in convs if c.args[6]]
    ok = not left and len(convs) == 1 and pg.CALLS.get("convolution_transposed") == {"rewritten": 1}
    check_result("ConvTranspose pass_graph", torch.tensor([ok]), torch.tensor([True]))
    return ok


EXTRA = {"one_d_no_bias": run_no_bias, "pass_graph": run_pass_graph}


if __name__ == "__main__":
    names = sys.argv[1:] or [*CASES, *EXTRA]
    for n in names:
        (EXTRA[n] if n in EXTRA else run_case)(n)
