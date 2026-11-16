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


if __name__ == "__main__":
    test_spec_rides_in_triton_meta_without_a_text_edit()
