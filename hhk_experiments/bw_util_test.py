import torch
import os
import sys


class VAdd(torch.nn.Module):
    def forward(self, x, y):
        return x + y


base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.append(base_dir)
os.environ["TOGSIM_CONFIG"] = f"{base_dir}/hhk_experiments/ndp_config.yml"
from Scheduler.scheduler import PyTorchSimRunner

device = PyTorchSimRunner.setup_device().custom_device()

input = torch.randn(1024, 1024).to(device=device)
weight = torch.randn(1024, 1024).to(device=device)

vadd = VAdd().to(device=device)
opt_fn = torch.compile(dynamic=False)(vadd)
npu_out = opt_fn(input, weight)

