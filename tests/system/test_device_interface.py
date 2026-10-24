"""What npu registers with Inductor, as Inductor reads it: the SM count is the machine's lane count
for any spelling of the device; Triton is present (through the interface, not a patch) and npu is a
Triton GPU; cpu keeps its own device op overrides; is_npu reads every spelling of the device; torch.npu's
RNG state reads and writes back for every spelling, its default included."""
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


def test_triton_and_gpu_type():
    """has_triton() is upstream's own function, answering True through NPUDeviceInterface."""
    import torch.utils._triton as triton_utils
    from torch._inductor.utils import is_gpu
    got = [triton_utils.has_triton.__module__ == "torch.utils._triton", triton_utils.has_triton(),
           is_gpu("npu")]
    check_result("TritonAndGpuType", torch.tensor(got), torch.ones(len(got), dtype=torch.bool))


def test_device_op_overrides():
    """npu gets its own overrides; cpu keeps Inductor's, which this backend no longer registers."""
    from torch._inductor.codegen.common import get_device_op_overrides
    from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides
    from pytorchsim.frontend.inductor.registration import NPUDeviceOpOverrides
    got = [isinstance(get_device_op_overrides("npu"), NPUDeviceOpOverrides),
           isinstance(get_device_op_overrides("cpu"), CpuDeviceOpOverrides)]
    check_result("DeviceOpOverrides", torch.tensor(got), torch.ones(len(got), dtype=torch.bool))


def test_is_npu_spellings():
    """is_npu reads a device, a device string with or without an index, and a tensor's device."""
    from pytorchsim.frontend.inductor import is_npu
    got = [is_npu("npu"), is_npu("npu:0"), is_npu(torch.device("npu", 0)),
           is_npu(torch.empty(1, device="npu")), not is_npu("cpu"), not is_npu(torch.empty(1))]
    check_result("IsNpuSpellings", torch.tensor(got), torch.ones(len(got), dtype=torch.bool))


def test_rng_state_for_every_spelling():
    """get_rng_state and set_rng_state take the default device, an index and the npu strings."""
    state = torch.npu.get_rng_state()
    for device in (0, "npu", "npu:0"):
        torch.npu.set_rng_state(state, device)
    torch.npu.set_rng_state(state)
    got = [bool(torch.equal(torch.npu.get_rng_state(*d), state)) for d in ((), (0,), ("npu",), ("npu:0",))]
    check_result("RngStateSpellings", torch.tensor(got), torch.ones(4, dtype=torch.bool))


if __name__ == "__main__":
    test_sm_count_is_the_lane_count()
    test_device_answers()
    test_triton_and_gpu_type()
    test_device_op_overrides()
    test_is_npu_spellings()
    test_rng_state_for_every_spelling()
