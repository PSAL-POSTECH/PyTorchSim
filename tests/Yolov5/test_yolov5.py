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
import shutil



def run_yolo(batch, config):
    device = torch.device("npu:0")

    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 128
    
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
    x = x.to(device)
    

    model.to(device)
    x = x.to(device)
    
    # Compile and run the model with PyTorchSim
    compiled_model = torch.compile(dynamic=False)(model)
    y = compiled_model(x)
    print("Yolo Simulation Done")


if __name__ == "__main__":

    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    config = os.environ.get(
        "TORCHSIM_CONFIG",
        default=f"{base_dir}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
    )
    args = argparse.ArgumentParser()
    args.add_argument("--batch", type=int, default=1)
    args.add_argument("--dump_path", type=str, default="results")
    args = args.parse_args()
    batch = args.batch

    run_yolo(batch, config)
