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
        # THE DEFAULT DEVICE IS PART OF THE LAUNCH, for this model, and scoped
        # to the compiled call rather than set for the process.
        #
        # Swinv2Layer.get_attn_mask builds its shifted-window mask with
        # `torch.zeros(...)` and no `device=`, then moves it with
        # `.to(hidden_states_windows.device)`. Building a constant on the host
        # and copying it once is an ordinary thing to do and costs nothing on a
        # GPU -- but torch.compile traces the whole forward, so that constant
        # becomes a CPU ISLAND inside the compiled graph and Inductor emits a
        # C++ kernel for it beside the device ones. Its CPU vectorizer then
        # fails to compile what it wrote (`decltype` of a scalar float, then
        # `Vectorized<float>::blendv`), which is an upstream defect reproducible
        # with stock torch and no PyTorchSim imported at all.
        #
        # Naming the default device removes the island instead of working
        # around it: the mask is built on the device, the backend compiles it
        # like any other elementwise work, and no C++ is generated to miscompile.
        #
        #     measured   26 kernels and a CPP compile error without this; 27
        #                kernels, cpp_fused = 0, and 4.77e-06 with it.
        #
        # `torch.device(...)` as a context manager is the documented scoped form
        # of `set_default_device`, so the CPU reference below is unaffected.
        with torch.device(device):
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
