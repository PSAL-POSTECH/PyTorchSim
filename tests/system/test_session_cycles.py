"""A `with torch.npu.TOGSimulator()` block prints, when it closes, one "[TOGSim] session simulated: N cycles, K launches"
line for every launch it ran, the line run_grid sums. Breaks when the block's cycles go unprinted, so a model test
that runs its device call in one session reports none."""
import os
import re
import subprocess
import sys

CHILD = """
import torch
dev = torch.device("npu:0")
x = torch.arange(1024, dtype=torch.float32).to(dev)
with torch.no_grad(), torch.npu.TOGSimulator():
    y = torch.compile(lambda t: t + 1)(x)
    z = torch.compile(lambda t: t * 2)(y)
assert torch.equal(z.cpu(), (torch.arange(1024, dtype=torch.float32) + 1) * 2)
"""


def test_a_block_prints_its_session_cycles():
    env = dict(os.environ, TORCHSIM_TIMING_MODE="1", TORCHSIM_FUNCTIONAL_MODE="1")
    run = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                         cwd=os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
    out = run.stdout + run.stderr
    assert run.returncode == 0, out[-3000:]
    lines = re.findall(r"\[TOGSim\] session simulated: (\d+) cycles, (\d+) launches", out)
    assert len(lines) == 1 and int(lines[0][0]) > 0 and int(lines[0][1]) >= 2, (lines, out[-3000:])


if __name__ == "__main__":
    test_a_block_prints_its_session_cycles()
    print("PASS")
