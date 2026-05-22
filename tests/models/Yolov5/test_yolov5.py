import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime

import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def run_yolo(batch, config):
    import copy

    device = torch.device("npu:0")

    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 128

    # Load model and prepare input
    model = torch.hub.load("ultralytics/yolov5", "yolov5s").cpu().eval()
    url = "https://ultralytics.com/images/zidane.jpg"

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    imgsz = 64
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])

    x = transform(img).unsqueeze(0)   # [1, 3, H, W]

    # CPU version
    model_cpu = copy.deepcopy(model).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    # NPU version
    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    # Compare results
    # YOLOv5 output is typically a list or tensor, handle both cases
    if isinstance(y_cpu, (list, tuple)):
        for i, (out_npu, out_cpu) in enumerate(zip(y_npu, y_cpu)):
            test_result(f"YOLOv5 Output {i}", out_npu, out_cpu)
    else:
        test_result("YOLOv5 Output", y_npu, y_cpu)

    print("Yolo Simulation Done")


def test_c3_module(device, batch=1, c1=64, c2=128, n=1, h=64, w=64):
    import copy
    import sys

    # Import C3 module from YOLOv5
    try:
        # Load model first to ensure hub cache is populated
        _ = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=False)

        # Try to import from torch hub cache
        hub_path = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if os.path.exists(hub_path):
            sys.path.insert(0, hub_path)
        # Import C3 module
        from models.common import C3  # noqa: F401
    except Exception as e:
        print(f"Warning: Could not import C3 module: {e}")
        print("Skipping C3 module test")
        return

    torch.manual_seed(0)

    # Create input tensor
    x = torch.randn(batch, c1, h, w)

    # CPU version
    model_cpu = C3(c1, c2, n=n, shortcut=True, g=1, e=0.5).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    # NPU version
    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    # Compare results
    if isinstance(y_cpu, (list, tuple)):
        for i, (out_npu, out_cpu) in enumerate(zip(y_npu, y_cpu)):
            test_result(f"C3 Output {i}", out_npu, out_cpu)
    else:
        test_result("C3 Output", y_npu, y_cpu)
    print("C3 Module Test Done")


def test_bottleneck_module(device, batch=1, c1=64, c2=64, shortcut=True, g=1, e=0.5, h=16, w=16):
    import copy
    import sys

    # Import Bottleneck module from YOLOv5
    try:
        # Load model first to ensure hub cache is populated
        _ = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=False)

        # Try to import from torch hub cache
        hub_path = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if os.path.exists(hub_path):
            sys.path.insert(0, hub_path)
        # Import Bottleneck module
        from models.common import Bottleneck  # noqa: F401
    except Exception as e:
        print(f"Warning: Could not import Bottleneck module: {e}")
        print("Skipping Bottleneck module test")
        return

    torch.manual_seed(0)

    # Create input tensor
    x = torch.randn(batch, c1, h, w)

    # CPU version
    model_cpu = Bottleneck(c1, c2, shortcut=shortcut, g=g, e=e).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    # NPU version
    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    # Compare results
    test_result("Bottleneck Module", y_npu, y_cpu)
    print("Bottleneck Module Test Done")


def test_conv_module(device, batch=1, c1=32, c2=64, k=3, s=1, p=None, g=1, d=1, act=True, h=16, w=16):
    import copy
    import sys

    # Import Conv module from YOLOv5
    try:
        # Load model first to ensure hub cache is populated
        _ = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=False)

        # Try to import from torch hub cache
        hub_path = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if os.path.exists(hub_path):
            sys.path.insert(0, hub_path)
        # Import Conv module
        from models.common import Conv  # noqa: F401
    except Exception as e:
        print(f"Warning: Could not import Conv module: {e}")
        print("Skipping Conv module test")
        return

    torch.manual_seed(0)

    # Create input tensor
    x = torch.randn(batch, c1, h, w)

    # CPU version
    model_cpu = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act).cpu().eval()
    x_cpu = copy.deepcopy(x).cpu()
    y_cpu = model_cpu(x_cpu)

    # NPU version
    model_npu = model_cpu.to(device).eval()
    x_npu = copy.deepcopy(x).to(device)
    compiled_model_npu = torch.compile(dynamic=False)(model_npu)
    y_npu = compiled_model_npu(x_npu)

    # Compare results
    test_result("Conv Module", y_npu, y_cpu)
    print("Conv Module Test Done")


def test_concat_4d(device):
    """
    Test concatenating 3 tensors along dimension 4
    Shapes: (1, 3, 4, 4, 2), (1, 3, 4, 4, 2), (1, 3, 4, 4, 81)
    Result: (1, 3, 4, 4, 85)
    """
    import copy

    torch.manual_seed(0)

    # Create 3 input tensors
    x1 = torch.ones(1, 3, 4, 4, 2)
    x2 = torch.ones(1, 3, 4, 4, 2) * 2
    x3 = torch.ones(1, 3, 4, 4, 81) * 3

    # CPU version
    x1_cpu = copy.deepcopy(x1).cpu()
    x2_cpu = copy.deepcopy(x2).cpu()
    x3_cpu = copy.deepcopy(x3).cpu()
    y_cpu = torch.cat([x1_cpu, x2_cpu, x3_cpu], dim=4)

    # NPU version
    x1_npu = copy.deepcopy(x1).to(device)
    x2_npu = copy.deepcopy(x2).to(device)
    x3_npu = copy.deepcopy(x3).to(device)

    def concat_fn(x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=4)

    compiled_concat = torch.compile(dynamic=False)(concat_fn)
    y_npu = compiled_concat(x1_npu, x2_npu, x3_npu)

    # Compare results
    test_result("Concat 4D", y_npu, y_cpu)
    print(f"Output shape: {y_npu.shape}")
    print("Concat 4D Test Done")

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

    # Test Concat 4D
    # print("=" * 80)
    # print("Testing Concat 4D")
    # print("=" * 80)
    # test_concat_4d(device)

    # Test Conv module
    # print("\n" + "=" * 80)
    # print("Testing Conv Module")
    # print("=" * 80)
    # test_conv_module(device, batch=batch, c1=32, c2=32, k=1, s=1, p=None, g=1, d=1, act=False, h=16, w=16)

    # Test Bottleneck module
    # print("\n" + "=" * 80)
    # print("Testing Bottleneck Module")
    # print("=" * 80)
    # test_bottleneck_module(device, batch=batch, c1=32, c2=32, shortcut=True, g=1, e=0.5, h=16, w=16)

    # Test C3 module
    # print("\n" + "=" * 80)
    # print("Testing C3 Module")
    # print("=" * 80)
    # test_c3_module(device, batch=batch, c1=64, c2=64, n=1, h=16, w=16)

    # Test full YOLOv5 model
    print("\n" + "=" * 80)
    print("Testing Full YOLOv5 Model")
    print("=" * 80)
    run_yolo(batch, config)
