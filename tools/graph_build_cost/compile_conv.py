"""Compile one conv2d through PyTorchSim and keep only the produced trace.so."""
import sys, torch, torch.nn.functional as F
dev = torch.device("npu:0")
b, ci, co, hw, k, st, pad = map(int, sys.argv[1:8])
torch.manual_seed(0)
x = torch.randn(b, ci, hw, hw).to(memory_format=torch.channels_last, device=dev)
w = torch.randn(co, ci, k, k).to(memory_format=torch.channels_last, device=dev)
with torch.no_grad():
    torch.compile(lambda x, w: F.conv2d(x, w, stride=st, padding=pad), dynamic=False)(x, w)
print("COMPILE_DONE")
