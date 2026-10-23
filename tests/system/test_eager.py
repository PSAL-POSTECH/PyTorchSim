import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

torch.npu.register_eager_to_compile(["aten::mul.Tensor", "aten::add.Tensor"])

if __name__ == "__main__":
    device = torch.device("npu:0")
    x_cpu = torch.ones(10, 10)
    y_cpu = torch.ones(10, 10)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    z = x * y
    z = x + z
    test_result("Eager mul + add", z, x_cpu + x_cpu * y_cpu)
