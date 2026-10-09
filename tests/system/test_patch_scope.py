"""The npu backend patches Inductor in place; nothing it patches may answer for
another device. Every check compares a cpu argument against upstream's own
answer, computed from the classes this backend subclasses rather than replaces."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.select_algorithm import AlgorithmSelectorCache


def test_reduction_split_is_upstream_off_device():
    """A cpu reduction keeps upstream's split factor, not this route's fixed 1."""
    device = torch.device("cpu")
    shapes = ((1 << 20, 2), (1 << 16, 1), (1 << 14, 4))
    ours = [config.inductor_choices_class().reduction_split_factor(
        device, r, n, True) for r, n in shapes]
    upstream = [InductorChoices().reduction_split_factor(
        device, r, n, True) for r, n in shapes]
    test_result("ReductionSplitOffDevice", torch.tensor(ours),
                torch.tensor(upstream))


def test_benchmark_operand_is_upstream_off_device():
    """A cpu autotune operand is still sampled, not this route's allocation."""
    ours = AlgorithmSelectorCache.generate_example_value(
        (64,), (1,), "cpu", torch.float32, 0)
    with preserve_rng_state():
        upstream = rand_strided((64,), (1,), device="cpu",
                                dtype=torch.float32, extra_size=0)
    test_result("BenchmarkOperandOffDevice", ours, upstream)


def test_device_predicate_reads_every_spelling():
    """Inductor hands the device as a string as often as a torch.device.

    A predicate that answers only torch.device sends the string form to
    upstream, whose operand is sampled -- one silent CPU fallback per call.
    """
    from PyTorchSimCodegen.inductor_patches import _is_npu

    spellings = ("npu", "npu:0", torch.device("npu"), torch.device("npu:0"),
                 "cpu", "cuda", torch.device("cpu"), None)
    got = torch.tensor([_is_npu(d) for d in spellings])
    test_result("DevicePredicateSpellings", got,
                torch.tensor([True, True, True, True, False, False, False, False]))


def test_conv_grid_is_upstream_off_device():
    """A grouped conv grid outside an npu graph keeps upstream's channel axis.

    Upstream grids all channels at once; this route grids one group's worth,
    which is 1 program here against upstream's 2.
    """
    import torch._inductor.kernel.conv as conv_kernel

    meta = {"BLOCK_M": 64, "BLOCK_N": 128, "GROUPS": 4}
    got = torch.tensor(list(conv_kernel.conv2d_grid(2, 256, 8, 8, meta))
                       + list(conv_kernel.conv3d_grid(2, 256, 4, 8, 8, meta)))
    test_result("ConvGridOffDevice", got, torch.tensor([2, 2, 4, 8, 2, 4]))


def test_this_device_still_gets_its_own_answers():
    """The scoping did not disable the route: npu keeps split 1 and allocation."""
    split = config.inductor_choices_class().reduction_split_factor(
        torch.device("npu"), 1 << 20, 2, True)
    operand = AlgorithmSelectorCache.generate_example_value(
        (64,), (1,), "npu", torch.float32, 0)
    got = torch.tensor([split, operand.numel(), operand.device.type == "npu"])
    test_result("OnDeviceAnswersKept", got, torch.tensor([1, 64, True]))


if __name__ == "__main__":
    test_reduction_split_is_upstream_off_device()
    test_benchmark_operand_is_upstream_off_device()
    test_device_predicate_reads_every_spelling()
    test_conv_grid_is_upstream_off_device()
    test_this_device_still_gets_its_own_answers()
