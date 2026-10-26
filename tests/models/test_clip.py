import argparse
import os
import sys

import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


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


# openai/clip-vit-base-patch32, vision_config as published; only num_hidden_layers is cut.
_CLIP_VIT_B32 = dict(
    hidden_size=768,
    intermediate_size=3072,
    projection_dim=512,
    num_attention_heads=12,
    num_channels=3,
    image_size=224,
    patch_size=32,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
)


def test_clip(device, batch=2, num_layers=2):
    """CLIP ViT-B/32 vision backbone at its published per-layer size, with batch > 1.
    The patch-embedding conv (kernel = stride = patch_size) routes to the multi-tile
    conv mapping, whose full-kernel tile overflows SPAD at batch > 1 -- that used to
    raise "Cannot find a valid mapping" (GitHub issue #252)."""
    torch.manual_seed(0)
    cfg = CLIPVisionConfig(**_CLIP_VIT_B32, num_hidden_layers=num_layers)
    image_size = cfg.image_size
    with torch.no_grad():
        model = CLIPVisionModel(cfg).eval()
        model.apply(_init_weights)

        x = torch.randn(batch, 3, image_size, image_size)
        out_cpu = model(pixel_values=x.cpu()).last_hidden_state

        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        out_device = opt_model(pixel_values=x.to(device)).last_hidden_state

    check_result(f"CLIP vision (batch={batch})", out_device, out_cpu)
    print("Max diff >", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("CLIP Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIP vision test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()
    test_clip(
        torch.device("npu:0"),
        batch=args.batch,
        num_layers=args.num_layers,
    )
