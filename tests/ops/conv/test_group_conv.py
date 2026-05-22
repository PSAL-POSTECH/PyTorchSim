import os
import sys
import torch
import torch._dynamo
from Simulator.simulator import TOGSimulator
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_group_convolution(
    device,
    groups=2,
    stride=1,
    padding=1,
    batch_size=2,
    c_per_group=8,
    out_per_group=12,
    spatial=16,
    kernel_size=3,
    seed=0,
):
    """``torch.compile`` on NPU vs CPU reference — same structure as ``test_matmul`` / ``test_conv2d``."""

    def custom_group_conv(a, weight, bias):
        return torch.convolution(
            a,
            weight,
            bias,
            (stride, stride),
            (padding, padding),
            (1, 1),
            False,
            (0, 0),
            groups,
        )

    torch.manual_seed(seed)
    c_in = c_per_group * groups
    c_out = out_per_group * groups
    k = kernel_size
    x = torch.randn(batch_size, c_in, spatial, spatial)
    wgt = torch.randn(c_out, c_in // groups, k, k)
    b = torch.randn(c_out)

    x1 = x.to(device=device, memory_format=torch.channels_last)
    w1 = wgt.to(device=device, memory_format=torch.channels_last)
    b1 = b.to(device=device)
    x2 = x.to("cpu", memory_format=torch.channels_last)
    w2 = wgt.to("cpu", memory_format=torch.channels_last)
    b2 = b.to("cpu")

    opt_fn = torch.compile(dynamic=False)(custom_group_conv)
    res = opt_fn(x1, w1, b1)
    y = custom_group_conv(x2, w2, b2)
    label = f"Group Conv Forward (groups={groups}, stride={stride}, pad={padding})"
    test_result(label, res, y, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - y)))


if __name__ == "__main__":
    device = torch.device("npu:0")
    with torch.no_grad():
        #test_group_convolution(device, batch_size=1, groups=2, stride=1, padding=1, seed=0)
        #test_group_convolution(device, batch_size=1, groups=4, stride=1, padding=1, seed=1)
        #test_group_convolution(device, batch_size=1, groups=2, stride=2, padding=1, seed=2)
        test_group_convolution(device, batch_size=1, groups=240, stride=2, padding=1, seed=2, c_per_group=1, out_per_group=1, spatial=40)

        #test_group_convolution(device, batch_size=1, groups=240, stride=2, padding=1, seed=2, c_per_group=1, out_per_group=1)
    print("test_group_conv_decomposition: all passed")
