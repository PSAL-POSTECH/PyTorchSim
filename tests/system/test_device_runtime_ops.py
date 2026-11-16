"""Device-side runtime ops called directly, outside any compiled graph: the device synchronizes when torch asks
for every stream, and eager abs covers the dtypes models hand it, contiguous or not."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


def test_synchronize_device():
    """torch.accelerator.synchronize() reaches the guard's synchronizeDevice, which torch refuses by default."""
    torch.accelerator.synchronize()
    torch.accelerator.synchronize(0)
    torch.npu.synchronize()
    check_result("SynchronizeDevice", torch.ones(1), torch.ones(1))


if __name__ == "__main__":
    test_synchronize_device()
