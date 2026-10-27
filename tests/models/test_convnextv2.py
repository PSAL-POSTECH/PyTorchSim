import argparse
import os
import sys

import torch
from transformers import ConvNextV2Config, ConvNextV2Model

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


# facebook/convnextv2-tiny-1k-224 as published; only depths are cut, [3, 3, 9, 3] -> [1, 1, 1, 1].
_CONVNEXTV2_TINY = dict(
    num_channels=3,
    num_stages=4,
    hidden_sizes=[96, 192, 384, 768],
    hidden_act="gelu",
    layer_norm_eps=1e-12,
    drop_path_rate=0.0,
    image_size=224,
    patch_size=4,
)

# (rtol, atol) per dtype; the bf16 atol is 3x the worst bf16 eager-vs-Inductor CPU diff at S = 128.
_TOL = {"bfloat16": (5e-2, 0.1), "float32": (1e-4, 1e-4)}
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def test_convnextv2(device, batch=2, depths=(1, 1, 1, 1), dtype="bfloat16"):
    """ConvNeXt V2 tiny at its published per-stage widths, with batch > 1, in `dtype` at its
    `_TOL`. Two things used to break here (GitHub issue #255): the channels-first LayerNorm reduces the
    contiguous axis, which the GEMM template's 2-D reduction-epilogue frame cannot
    express, and the depthwise 7x7 conv lowers to gather kernels whose index-expression
    scratch buffer exhausted the SPAD budget."""
    torch.manual_seed(0)
    cfg = ConvNextV2Config(**_CONVNEXTV2_TINY, depths=list(depths))
    image_size = cfg.image_size
    with torch.no_grad():
        model = ConvNextV2Model(cfg).eval()
        model.apply(_init_weights)
        model.to(_DTYPES[dtype])

        x = torch.randn(batch, 3, image_size, image_size).to(_DTYPES[dtype])
        out_cpu = model(pixel_values=x.cpu()).last_hidden_state

        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        out_device = opt_model(pixel_values=x.to(device)).last_hidden_state

    check_result(f"ConvNeXt V2 (batch={batch})", out_device, out_cpu, *_TOL[dtype])
    print("Max diff >", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("ConvNeXt V2 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvNeXt V2 test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(_DTYPES))
    args = parser.parse_args()
    test_convnextv2(
        torch.device("npu:0"),
        batch=args.batch,
        dtype=args.dtype,
    )
