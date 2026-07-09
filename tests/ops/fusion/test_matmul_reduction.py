import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_matmul_reduce(device, M=512, N=512, K=512):
    def matmul_fused(a, b):
        result = torch.matmul(a, b)
        return result, result.max(dim=-2).values
    torch.manual_seed(0)
    input = torch.randn(M, K)
    weight = torch.randn(K, N)
    #input = torch.arange(1, M * K + 1, dtype=torch.float32).reshape(M, K).to(dtype=torch.float32)
    #weight = torch.eye(K, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1)
    y = matmul_fused(x2, w2)
    test_result("Matmul Reduction Fusion activation", res[0], y[0])
    test_result("Matmul Reduction Fusion reduction", res[1], y[1])

def test_matmul_reduce_last_dim(device, M=256, N=96, K=96):
    """Reducing the matmul output's contiguous (last) axis cannot be expressed in the GEMM's
    2-D reduction-epilogue frame, so the reduction has to run as its own kernel. Fusing it
    anyway silently produces wrong values, so this only checks the numbers."""
    def matmul_fused(a, b):
        result = torch.matmul(a, b)
        return result - result.mean(dim=-1, keepdim=True)
    torch.manual_seed(0)
    input = torch.randn(M, K)
    weight = torch.randn(K, N)
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(input.to(device=device), weight.to(device=device))
    y = matmul_fused(input.to("cpu"), weight.to("cpu"))
    test_result("Matmul reduce over contiguous axis", res, y)

def test_matmul_var_mean(device, size=512):
    def matmul_fused(a, b, c):
        result = torch.matmul(a, b.T)
        var, mean = torch.var_mean(result, dim=-2)
        return result, var, mean
    torch.manual_seed(0)
    N = size
    input = torch.randn(1024, 768)
    weight = torch.randn(512, 768)
    #input = torch.arange(1, N * N + 1, dtype=torch.float32).reshape(N, N).to(dtype=torch.float32)
    #weight = torch.eye(N, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    c = 7
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, c)
    y = matmul_fused(x2, w2, c)
    test_result("Matmul var_mean Fusion activation", res[0], y[0])
    test_result("Matmul var_mean Fusion reduction", res[1], y[1])
    test_result("Matmul var_mean Fusion reduction", res[2], y[2])

def test_matmul_add_var_mean(device, M=768, N=512, K=3072):
    def matmul_fused(a, b, c, d):
        result = torch.matmul(a, b.T) + c.T
        var, mean = torch.var_mean(result + d, dim=-2)
        return result, var, mean
    torch.manual_seed(0)
    input = torch.randn(M, K)
    weight = torch.randn(N, K)
    bias = torch.zeros(N, M)
    residual = torch.randn(M,N)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    r1 = residual.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    r2 = residual.to("cpu")
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, b1, r1)
    y = matmul_fused(x2, w2, b2, r2)
    test_result("Matmul+residual+var_mean Fusion activation", res[0], y[0])
    test_result("Matmul+residual+var_mean Fusion reduction", res[1], y[1])
    test_result("Matmul+residual+var_mean Fusion reduction", res[2], y[2])

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_matmul_reduce(device, 3072, 512, 768)
    test_matmul_reduce_last_dim(device)
    test_matmul_var_mean(device)
    test_matmul_add_var_mean(device)
