"""A kernel timed alone (template_timing._time_tile, a fusion's benchmark) whose TOG reads runtime attributes is
timed on the records a Spike run on zeros leaves, when none of them reads an input: an attention output's
transpose is finite, not inf; records that read an input are refused by name, and a kernel with none reads none."""
import math
import os
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim import contract  # noqa: E402
from pytorchsim.backend import template_timing  # noqa: E402

H, S, D = 4, 64, 128
N = H * S * D
SRC = "\nimport triton\nimport triton.language as tl\n\nfrom torch._inductor.runtime import triton_helpers, triton_heuristics\nfrom torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\nfrom torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties\ntriton_helpers.set_driver_to_gpu()\n\n@triton_heuristics.pointwise(\n    size_hints={'x': 32768}, \n    filename=__file__,\n    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=128, cc=0, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},\n    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 131072}, 'backend_hash': '0B15F82B81FCD8E04E3E74AACBED69EC5A670725CA2756E78298198724CAAD77', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},\n    min_elem_per_thread=0\n)\n@triton.jit\ndef triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):\n    xnumel = 32768\n    xoffset = tl.program_id(0) * XBLOCK\n    xindex = xoffset + tl.arange(0, XBLOCK)[:]\n    xmask = tl.full([XBLOCK], True, tl.int1)[:]\n    x0 = (xindex % 128)\n    x1 = ((xindex // 128) % 4)\n    x2 = xindex // 512\n    x3 = xindex\n    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 8192*x1), None).to(tl.float32)\n    tl.store(out_ptr0 + (x3), tmp0, None)\n"
META = {"kernel_name": "triton_", "signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16", "xnumel": "i32",
                                                "XBLOCK": "constexpr"},
        "constants": {}, "call_params": ["in_ptr0", "out_ptr0"], "numels": {"xnumel": N, "r0_numel": 1},
        "inside_reduction": False, "fixed_config": {"XBLOCK": N}, "template_grid": None, "graph_prefetch": [],
        "args": [{"name": "in_ptr0", "role": "in", "buffer": "buf0", "dtype": "bfloat16", "numel": N,
                  "sizes": [1, H, S, D]},
                 {"name": "out_ptr0", "role": "out", "buffer": "buf1", "dtype": "bfloat16", "numel": N,
                  "sizes": [1, S, H, D]}]}


def test_a_transpose_whose_records_read_no_input_is_timed():
    with tempfile.TemporaryDirectory() as d:
        cycles = template_timing._time_tile(SRC, dict(META), "triton_", d)
        attributes = contract.kernel_object(d)["runtime_attributes"]["attributes"]
        assert attributes and not any(a["sources"] for a in attributes), attributes
        assert math.isfinite(cycles) and cycles > 0, cycles


def test_records_that_read_an_input_are_refused_by_name():
    manifest = {"runtime_attributes": {"attributes": [{"id": 0, "sources": ["in_ptr0"]}]}}
    with mock.patch.object(contract, "kernel_object", return_value=manifest):
        try:
            template_timing._records("unused", META)
        except RuntimeError as e:
            assert "in_ptr0" in str(e), e
            return
    raise AssertionError("records that read an input were not refused")


def test_a_kernel_with_no_records_reads_none():
    with mock.patch.object(contract, "kernel_object", return_value={"runtime_attributes": {}}):
        assert template_timing._records("unused", META) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("Bench Runtime Attributes Test Passed")
