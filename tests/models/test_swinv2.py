import argparse
import os
import sys

import torch
from transformers import Swinv2Config, Swinv2Model

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


# microsoft/swinv2-tiny-patch4-window8-256 as published; only depths are cut, [2, 2, 6, 2] -> [2, 2, 2, 2].
_SWINV2_TINY_W8 = dict(
    image_size=256,
    patch_size=4,
    num_channels=3,
    embed_dim=96,
    num_heads=[3, 6, 12, 24],
    window_size=8,
    pretrained_window_sizes=[0, 0, 0, 0],
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    encoder_stride=32,
)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.5), "float32": (1e-4, 1e-4)}
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def test_swinv2(device, batch=2, depths=(2, 2, 2, 2), dtype="bfloat16"):
    """SwinV2 tiny (window 8, 256 px), all four stages at their published widths, with
    batch > 1, in `dtype` at its `_TOL`. Two blocks per stage put a SHIFTED-window block
    (torch.roll composed with window partition/reverse) in each of the first three stages
    -- the path that used to fail codegen with "Unlinearized floor/mod in DMA index"
    (GitHub issue #251).

    The compiled call runs under `torch.device(device)`: Swinv2Layer.get_attn_mask builds
    its shifted-window mask with `torch.zeros(...)` and no `device=`, so without a default
    device that constant is a CPU island inside the compiled graph and Inductor emits a C++
    kernel for it, which its own CPU vectorizer fails to compile (`decltype` of a scalar
    float, then `Vectorized<float>::blendv`; reproducible with stock torch). Under the
    scoped default device the mask is built on the device and compiled like any other
    elementwise work. The context manager is the scoped form of `set_default_device`, so
    the CPU reference is unaffected."""
    torch.manual_seed(0)
    cfg = Swinv2Config(**_SWINV2_TINY_W8, depths=list(depths))
    image_size = cfg.image_size
    with torch.no_grad():
        model = Swinv2Model(cfg).eval()
        model.apply(_init_weights)
        model.to(_DTYPES[dtype])

        x = torch.randn(batch, 3, image_size, image_size).to(_DTYPES[dtype])
        x_device = x.to(device=device)
        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        with torch.device(device):
            out_device = opt_model(pixel_values=x_device).last_hidden_state

        out_cpu = model.cpu()(pixel_values=x.cpu()).last_hidden_state

    check_result(f"SwinV2 shifted-window (batch={batch})", out_device, out_cpu, *_TOL[dtype])
    print("Max diff >", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("SwinV2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SwinV2 shifted-window test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(_DTYPES))
    args = parser.parse_args()
    test_swinv2(
        torch.device("npu:0"),
        batch=args.batch,
        dtype=args.dtype,
    )
