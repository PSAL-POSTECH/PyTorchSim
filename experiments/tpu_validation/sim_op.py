"""One shape of one op through PyTorchSim on a timing config: the program the TPU table measured,
with fp16 standing in for the TPU's bf16 (same bytes).

    python sim_op.py gemm M K N       torch.matmul(a, b), fp16 in and out
    python sim_op.py softmax R C      torch.softmax(x, -1, dtype=float32).to(float16), fp16 in

TORCHSIM_TARGET must name the machine file (run.py sets it to the tag's config.yml).
"""
import os
import sys

sys.path.insert(0, os.environ["TORCHSIM_DIR"])
import torch  # noqa: E402

op, shape = sys.argv[1], tuple(map(int, sys.argv[2:]))
if "TORCHSIM_TARGET" not in os.environ:
    raise SystemExit("TORCHSIM_TARGET is not set: run.py sets it to the tag's config.yml")
torch.manual_seed(0)

if op == "gemm":
    M, K, N = shape
    inputs = (torch.randn(M, K, dtype=torch.float16), torch.randn(K, N, dtype=torch.float16))
    fn = lambda a, b: torch.matmul(a, b)  # noqa: E731
elif op == "softmax":
    R, C = shape
    inputs = ((torch.randn(R, C) * 3).to(torch.float16),)
    fn = lambda x: torch.softmax(x, dim=-1, dtype=torch.float32).to(torch.float16)  # noqa: E731
else:
    raise SystemExit(f"unknown op {op}")

device = torch.device("npu:0")
inputs = [t.to(device=device) for t in inputs]
opt_fn = torch.compile(dynamic=False)(fn)
with torch.npu.TOGSimulator(), torch.no_grad():
    torch.npu.launch_model(opt_fn, *inputs, stream_index=0, timestamp=0)
    torch.npu.synchronize()
print(f"{op} {'x'.join(map(str, shape))} Simulation Done")
