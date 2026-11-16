"""PyTorchSim attaches to Inductor through its public extension points only (PTB rule 16); one check per
attachment that used to be a text edit or a patch, each failing on the tree that had it."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result


def test_spec_rides_in_triton_meta_without_a_text_edit():
    """A generated kernel and a template both reach define_kernel with the spec already in the decorator's
    triton_meta, rendered by Inductor itself, and no function edits the source."""
    from pytorchsim.frontend.inductor.codegen import kernel_metadata
    from pytorchsim.frontend.inductor.codegen.scheduling import TritonNPUScheduling

    seen = []
    orig = TritonNPUScheduling.define_kernel

    def define(self, src_code, node_schedule, kernel):
        seen.append((src_code, kernel.triton_meta))
        return orig(self, src_code, node_schedule, kernel)

    TritonNPUScheduling.define_kernel = define
    try:
        device = torch.device("npu:0")
        torch.compile(dynamic=False)(lambda a, b: a + b)(torch.randn(64, 64).to(device),
                                                         torch.randn(64, 64).to(device))
        torch.compile(dynamic=False)(torch.mm)(torch.randn(128, 128).to(device),
                                               torch.randn(128, 128).to(device))
    finally:
        TritonNPUScheduling.define_kernel = orig
    carried = [("pytorchsim_spec" in src) and ("backend_options" in meta) for src, meta in seen]
    check_result("SpecInRenderedMeta", torch.tensor([hasattr(kernel_metadata, "apply_spec_to_source"),
                                                     len(seen) >= 2, all(carried)]),
                 torch.tensor([False, True, True]))


def test_selection_and_conv_are_inductors_own_but_for_the_operand_and_grid():
    """Inductor's selection (__call__, benchmark_choices), its convolution lowering and its conv grid functions are
    its own objects; the npu's choices class is importable under its own name without a rename."""
    import inspect

    import torch._inductor.kernel.conv as conv_kernel
    from torch._inductor import lowering
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    from pytorchsim.frontend.inductor import registration

    inductor = "torch._inductor"
    own = [AlgorithmSelectorCache.__call__.__module__.startswith(inductor),
           AlgorithmSelectorCache.benchmark_choices.__func__.__module__.startswith(inductor),
           inductor in lowering.lowerings[torch.ops.aten.convolution.default].__code__.co_filename.replace("/", "."),
           conv_kernel.conv2d_grid.fn.__module__.startswith(inductor),
           conv_kernel.conv3d_grid.fn.__module__.startswith(inductor),
           "__qualname__ =" not in inspect.getsource(registration)]
    check_result("SelectionAndConvAreInductors", torch.tensor(own), torch.tensor([True] * len(own)))


def test_conv_groups_are_read_off_the_node_being_lowered():
    """The conv heuristic sees a grouped convolution's groups through V.graph's node, and 1 outside a lowering."""
    from pytorchsim.frontend.inductor.autotune import template_kernel_configs

    check_result("ConvGroupsOutsideLowering", torch.tensor([template_kernel_configs.conv_groups_now()]),
                 torch.tensor([1]))


if __name__ == "__main__":
    test_spec_rides_in_triton_meta_without_a_text_edit()
    test_selection_and_conv_are_inductors_own_but_for_the_operand_and_grid()
    test_conv_groups_are_read_off_the_node_being_lowered()
