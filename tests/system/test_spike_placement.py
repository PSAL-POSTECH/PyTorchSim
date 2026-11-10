"""A launch's runtime/ holds the tog_tensor_addrs.txt its TOGSim launch directory does, so the wrapper Spike
runs puts every tensor where TOGSim puts it. Breaks when functional.write_inputs stops writing it, or writes
lines of its own instead of placement.write_tensor_addrs's."""
import os
import sys

os.environ["TORCHSIM_TIMING_MODE"] = "1"
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402

from pytorchsim import contract  # noqa: E402
from test_tensor_placement import ADDRS, _launch_dirs  # noqa: E402


def test_spike_is_given_the_addresses_togsim_is():
    device = torch.device("npu:0")
    a = torch.randn(64, 256).to(device)
    view = torch.randn(96, 256).to(device)[32:]
    read = torch.compile(lambda a, v: a * 2.0 + v, dynamic=False)
    with torch.no_grad(), torch.npu.TOGSimulator() as sim:
        y = read(a, view)
    torch.testing.assert_close(y.cpu(), a.cpu() * 2.0 + view.cpu())
    (launch,) = _launch_dirs(sim.trace_path)
    runtime = contract.runtime_dir(os.path.dirname(launch))
    with open(os.path.join(launch, ADDRS)) as f:
        togsim = f.read()
    with open(os.path.join(runtime, ADDRS)) as f:
        spike = f.read()
    assert spike == togsim, (spike, togsim)


if __name__ == "__main__":
    test_spike_is_given_the_addresses_togsim_is()
    print("PASS")
