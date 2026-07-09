import argparse
import os
import sys

import torch
from transformers import ConvNextV2Config, ConvNextV2Model

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def _init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        if m.weight is not None:
            torch.nn.init.ones_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def test_convnextv2(device, batch=2, image_size=64, patch_size=4):
    """ConvNeXt V2 with batch > 1. Two things used to break here (GitHub issue #255):
    the channels-first LayerNorm reduces the contiguous axis, which the GEMM template's
    2-D reduction-epilogue frame cannot express, and the depthwise 7x7 conv lowers to
    gather kernels whose index-expression scratch buffer exhausted the SPAD budget."""
    torch.manual_seed(0)
    cfg = ConvNextV2Config(
        depths=[1, 1, 1, 1],
        hidden_sizes=[96, 192, 384, 768],
        image_size=image_size,
        patch_size=patch_size,
    )
    with torch.no_grad():
        model = ConvNextV2Model(cfg).eval()
        model.apply(_init_weights)

        x = torch.randn(batch, 3, image_size, image_size)
        out_cpu = model(pixel_values=x.cpu()).last_hidden_state

        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        out_device = opt_model(pixel_values=x.to(device)).last_hidden_state

    test_result(f"ConvNeXt V2 (batch={batch})", out_device, out_cpu)
    print("Max diff >", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("ConvNeXt V2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvNeXt V2 test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    args = parser.parse_args()
    test_convnextv2(
        torch.device("npu:0"),
        batch=args.batch,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
