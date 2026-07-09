"""Compile one matmul through PyTorchSim and keep only the produced trace.so."""
import sys, torch
dev = torch.device("npu:0")
M, K, N = map(int, sys.argv[1:4])
torch.manual_seed(0)
a = torch.randn(M, K, device=dev); b = torch.randn(K, N, device=dev)
with torch.no_grad():
    torch.compile(lambda a, b: a @ b, dynamic=False)(a, b)
print("COMPILE_DONE")
