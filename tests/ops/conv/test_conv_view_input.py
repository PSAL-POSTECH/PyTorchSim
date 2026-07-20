import os
import sys
import torch
import torch._dynamo
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_conv_view_input(device, batch_size=1, channels=32, height=8, width=8, out_channels=16, kernel_size=2):
    """Conv whose input is a free view (ReinterpretView) over a lower-rank buffer.

    A contiguous (B, H*W, C) buffer already is the channels_last layout of (B, C, H, W),
    so permute+reshape is a pure view: inductor inserts no materializing copy and the conv
    wrapper is handed the raw 3D base buffer. The wrapper must rebuild the 4D logical input
    from the layout codegen assumed instead of reading X.shape. This is SegFormer's
    efficient-attention spatial-reduction conv.
    """
    def conv_on_view(a, b, bias):
        x = a * 2.0
        h = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        return torch.nn.functional.conv2d(h, b, bias, stride=kernel_size, padding=0)
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, height * width, channels)
    conv_kernel = torch.randn(out_channels, channels, kernel_size, kernel_size)
    conv_bias = torch.randn(out_channels)
    out = conv_on_view(conv_input, conv_kernel, conv_bias)
    opt_fn = torch.compile(dynamic=False)(conv_on_view)
    res = opt_fn(conv_input.to(device=device), conv_kernel.to(device=device), conv_bias.to(device=device))
    test_result("Conv2d view input", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

if __name__ == "__main__":
    device = torch.device("npu:0")
    torch._dynamo.config.cache_size_limit = 64
    with torch.no_grad():
        test_conv_view_input(device)
        test_conv_view_input(device, batch_size=1, channels=64, height=16, width=16, out_channels=32, kernel_size=4)
