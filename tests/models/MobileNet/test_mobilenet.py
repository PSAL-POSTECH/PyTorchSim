import sys
import argparse
import copy
import os

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torchvision.models import mobilenet_v2
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result



def _mobilenet_v2():
    try:
        from torchvision.models import MobileNet_V2_Weights

        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).cpu().eval()
    except Exception:
        return mobilenet_v2().cpu().eval()


def run_mobilenet(batch, config):
    device = torch.device("npu:0")

    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 128

    model = _mobilenet_v2()
    imgsz = 224
    x = torch.randn(batch, 3, imgsz, imgsz)

    model_cpu = copy.deepcopy(model).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    if isinstance(y_cpu, (list, tuple)):
        for i, (out_npu, out_cpu) in enumerate(zip(y_npu, y_cpu)):
            test_result(f"MobileNet Output {i}", out_npu, out_cpu)
    else:
        test_result("MobileNet Output", y_npu, y_cpu)

    print("MobileNet Simulation Done")


def test_inverted_residual_module(device, batch=1, inp=32, oup=32, stride=1, expand_ratio=6, h=28, w=28):
    from torchvision.models.mobilenetv2 import InvertedResidual

    torch.manual_seed(0)

    x = torch.randn(batch, inp, h, w)

    model_cpu = InvertedResidual(inp, oup, stride, expand_ratio).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    test_result("InvertedResidual Module", y_npu, y_cpu)
    print("InvertedResidual Module Test Done")


if __name__ == "__main__":
    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    config = os.environ.get(
        "TOGSIM_CONFIG",
        default=f"{base_dir}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
    )
    args = argparse.ArgumentParser()
    args.add_argument("--batch", type=int, default=1)
    args.add_argument("--dump_path", type=str, default="results")
    args = args.parse_args()
    batch = args.batch

    device = torch.device("npu:0")

    # print("\n" + "=" * 80)
    # print("Testing InvertedResidual Module")
    # print("=" * 80)
    # test_inverted_residual_module(device, batch=batch, inp=32, oup=32, stride=1, expand_ratio=6, h=28, w=28)

    print("\n" + "=" * 80)
    print("Testing Full MobileNet V2 Model")
    print("=" * 80)
    run_mobilenet(batch, config)
