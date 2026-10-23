"""npu's DeviceInterface as Inductor reads it: the device properties take the machine's lane count
as the SM count, for any spelling of the device, and Inductor's template gate sees that count."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.runtime.hints import DeviceProperties

from pytorchsim import config


def test_sm_count_is_the_lane_count():
    """Every spelling of the device answers, and the SM count is the machine's vector lanes."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    interface = get_interface_for_device("npu")
    counts = [interface.get_device_properties(d).multi_processor_count
              for d in (None, 0, "npu", "npu:0", torch.device("npu", 0))]
    counts.append(DeviceProperties.create(torch.device("npu", 0)).multi_processor_count)
    check_result("SmCountIsLaneCount", torch.tensor(counts), torch.full((len(counts),), lanes))


def test_device_answers():
    """Available, Triton-capable, bf16-capable, no CUDA compute capability."""
    interface = get_interface_for_device("npu")
    got = [interface.is_available(), interface.is_triton_capable(),
           interface.is_bf16_supported(), interface.get_compute_capability() == 0]
    check_result("DeviceAnswers", torch.tensor(got), torch.ones(len(got), dtype=torch.bool))


if __name__ == "__main__":
    test_sm_count_is_the_lane_count()
    test_device_answers()
