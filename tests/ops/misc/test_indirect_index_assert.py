"""A gather index out of range stops the launch with Inductor's own message: assert_indirect_indexing is on, so
the kernel's tl.device_assert reaches PTB, which stops Spike on it. Breaks when the flag is off (the gather reads
past its table and returns a value) or when PTB cannot lower a tensor tt.assert (the compile fails instead)."""
import os
import sys

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))

import torch  # noqa: E402
from result_check import check_result  # noqa: E402


def _gather(table, index):
    return table[index] * 2.0


def test_in_range_index_gathers(device):
    torch.manual_seed(0)
    table, index = torch.randn(64, 128), torch.randint(0, 64, [32])
    got = torch.compile(dynamic=False)(_gather)(table.to(device), index.to(device))
    check_result("in-range gather", got.cpu(), _gather(table, index))


def test_out_of_range_index_stops_the_launch(device):
    table, index = torch.randn(64, 128), torch.randint(0, 64, [32])
    index[5] = 64
    try:
        torch.compile(dynamic=False)(_gather)(table.to(device), index.to(device))
    except Exception as e:  # noqa: BLE001
        assert "index out of bounds" in str(e), str(e)[-600:]
        print("Test Passed: the launch stopped with", [l for l in str(e).splitlines() if "index out of bounds" in l][:1])
        return
    raise AssertionError("an index of 64 into a table of 64 rows did not stop the launch")


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_in_range_index_gathers(device)
    test_out_of_range_index_stops_the_launch(device)
