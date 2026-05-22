import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_matmul_vector(device, size=[56, 78, 239], dim=0):
    def matmul_fused(a, b, c, d):
        return torch.matmul(a, b) + c + d
    torch.manual_seed(0)
    input = torch.randn(size[:2])
    weight = torch.randn(size[1:])
    output_sz = [size[0], size[2]]
    output_sz[dim]=1
    bias = torch.zeros(output_sz)
    add = torch.zeros(output_sz)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    a1 = add.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    a2 = add.to("cpu")
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, a1, b1)
    y = matmul_fused(x2, w2, a2, b2)
    test_result("Matmul Vector Fusion Forward", res, y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_matmul_vector(device, size=[253, 123, 47], dim=0)
    test_matmul_vector(device, size=[253, 123, 47], dim=1)