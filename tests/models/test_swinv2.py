import argparse
import os
import sys

import torch
from transformers import Swinv2Config, Swinv2Model

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


def test_swinv2(device, batch=2, image_size=64, window_size=8):
    """SwinV2 backbone with batch > 1 exercises the SHIFTED-window attention (torch.roll
    composed with window partition/reverse) -- the path that used to fail codegen with
    "Unlinearized floor/mod in DMA index" (GitHub issue #251)."""
    torch.manual_seed(0)
    cfg = Swinv2Config(
        image_size=image_size,
        patch_size=4,
        embed_dim=96,
        depths=[2],            # 2 blocks -> block 1 is the shifted-window block
        num_heads=[3],
        window_size=window_size,
    )
    with torch.no_grad():
        model = Swinv2Model(cfg).eval()
        model.apply(_init_weights)

        x = torch.randn(batch, 3, image_size, image_size)
        x_device = x.to(device=device)
        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        out_device = opt_model(pixel_values=x_device).last_hidden_state

        out_cpu = model.cpu()(pixel_values=x.cpu()).last_hidden_state

    test_result(f"SwinV2 shifted-window (batch={batch})", out_device, out_cpu)
    print("Max diff >", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("SwinV2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SwinV2 shifted-window test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=8)
    args = parser.parse_args()
    test_swinv2(
        torch.device("npu:0"),
        batch=args.batch,
        image_size=args.image_size,
        window_size=args.window_size,
    )
