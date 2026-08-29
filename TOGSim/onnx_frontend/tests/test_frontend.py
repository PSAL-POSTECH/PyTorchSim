import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from TOGSim.acs import cycles as acs_cycles
from TOGSim.acs.node_mapping import ELEMENTWISE, KernelDesc, NodeDesc, Op, lookup
from TOGSim.onnx_frontend import emit
from TOGSim.onnx_frontend.config import array_dim_from_name
from TOGSim.onnx_frontend.onnx_ops import OPS, resolve
from TOGSim.onnx_frontend.tiling import Hardware, check_systolic, systolic_tile

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "example")
HW = Hardware(sa=8, vlane=8, vlen_bits=256)


def test_retarget_rewrites_only_the_named_constant():
    src = "static const int64_t M = 128;\nstatic const int64_t TILES = (M + 1) / 2;\n"
    out = emit.retarget(src, {"M": 512})
    assert "M = 512;" in out
    assert "(M + 1) / 2" in out          # derived constants follow, untouched


def test_retarget_refuses_a_constant_the_kernel_does_not_declare():
    # silently skipping leaves the kernel at the example's size, and the run
    # still reports a cycle count -- for a shape nobody asked about
    with pytest.raises(KeyError):
        emit.retarget("static const int64_t M = 128;\n", {"TM": 32})


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))


def test_every_mapped_kernel_exists_and_declares_its_constants(op, kernels):
    for kernel in kernels:
        path = os.path.join(EXAMPLE, kernel.directory, kernel.stem + ".cpp")
        assert os.path.exists(path), f"{op}: no kernel at {path}"
        source = open(path).read()
        for name in kernel.size_names + kernel.tile_names + kernel.hw_names:
            # the kernels align their '=' signs, so match on the declaration
            assert re.search(rf"\b{re.escape(name)}\s*=", source), \
                f"{op}/{kernel.stem}: no constant {name}"


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))


def test_every_mapped_kernel_has_a_node_table(op, kernels):
    for kernel in kernels:
        lookup(kernel.stem)              # raises with a message if absent


def test_a_fused_node_decomposes_to_kernels_that_exist():
    assert [k.stem for k in resolve("BiasGelu")] == ["bias_act", "bias_gelu"]
    assert [k.stem for k in resolve("SkipLayerNormalization")] == \
        ["bias_act", "layernorm"]


def test_a_fused_node_with_its_own_kernel_maps_straight_to_it():
    assert [k.stem for k in resolve("Attention")] == ["attention"]


def test_an_unsupported_op_is_reported_not_guessed():
    assert resolve("Einsum") is None     # None -> unmapped, never costed


def test_metadata_ops_carry_no_compute():
    assert resolve("Reshape") == []


def test_systolic_tile_is_a_multiple_of_the_array_geometry():
    tm, tn, tk = systolic_tile(512, 1024, 512, HW)
    assert tm % HW.sa == 0
    # under SA*VLANE the kernel's STEPS_N truncates to 0 and issues no matmul
    assert tn % (HW.sa * HW.vlane) == 0 and tn >= HW.sa * HW.vlane


def test_too_narrow_for_the_array_is_refused():
    check_systolic(64, HW)               # 64 == SA*VLANE, the minimum
    with pytest.raises(ValueError):
        check_systolic(63, HW)


def test_array_dim_comes_from_the_config_name():
    assert array_dim_from_name("systolic_ws_8x8_c1_simple_noc.yml") == 8
    assert array_dim_from_name("systolic_ws_128x128_c2_booksim_tpuv3.yml") == 128
    assert array_dim_from_name("no_array_here.yml") is None


def test_the_table_column_is_an_overlap_not_an_interval():
    rows = acs_cycles.build_table(lookup("gemm"),
                                  {"rows": 32, "elems": 32 * 256}, HW)
    preload_cycles, preload_overlap = rows[1]
    assert preload_cycles - preload_overlap == 0

    matmul_cycles, matmul_overlap = rows[2]
    assert matmul_cycles - matmul_overlap == min(HW.sa, 32)


def test_vector_work_is_charged_in_full():
    rows = acs_cycles.build_table(lookup("bias_act"), {"elems": 4096}, HW)
    cycles, overlapping = rows[0]
    assert overlapping == 0
    assert cycles == (4096 // HW.throughput) * 2      # + bias, then max(x, 0)


def test_gelu_costs_more_than_relu_on_the_same_tile():
    tile = {"elems": 4096}
    relu = acs_cycles.build_table(lookup("bias_act"), tile, HW)[0][0]
    gelu = acs_cycles.build_table(lookup("bias_gelu"), tile, HW)[0][0]
    assert gelu == 3 * relu              # six instructions against two


def test_the_cost_function_can_be_replaced():
    desc = KernelDesc("x", (NodeDesc("n", "vector", (Op(ELEMENTWISE, 1, "op"),)),))
    try:
        acs_cycles.set_cost_function(lambda node, tile, hw: (99, 0))
        assert acs_cycles.build_table(desc, {"elems": 1}, HW) == [(99, 99)]
    finally:
        acs_cycles.set_cost_function(acs_cycles.default_cost)


def _kernel_source(stem):
    import glob
    hits = glob.glob(os.path.join(EXAMPLE, "*", stem + ".cpp"))
    return hits[0] if hits else None


@pytest.mark.parametrize("stem", sorted(
    k for k in __import__("TOGSim.acs.node_mapping", fromlist=["KERNELS"]).KERNELS))


def test_table_rows_match_the_kernel_and_the_node_list(stem):
    import re
    from TOGSim.acs.node_mapping import KERNELS

    source = _kernel_source(stem)
    if source is None:
        pytest.skip(f"{stem} has no kernel in example/")

    text = open(source).read()
    tids = set(re.findall(r"static const uint64_t (TID_\w+)\s*=", text))

    nodes = KERNELS[stem].nodes
    assert len(tids) == len(nodes), (
        f"{stem}: {len(tids)} tile_ids in the .cpp but {len(nodes)} in "
        f"node_mapping.py")

    # a table is only present when one was generated or measured; when it is,
    # its row count has to match too
    table = os.path.join(os.path.dirname(source), "cycles.tsv")
    if os.path.exists(table):
        rows = [l for l in open(table)
                if l.strip() and not l.startswith("#")]
        assert len(rows) == len(nodes), (
            f"{stem}: cycles.tsv has {len(rows)} rows but {len(nodes)} nodes")


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))


def test_one_operator_maps_to_one_kernel(op, kernels):
    assert len(kernels) == 1, (
        f"{op} maps to {[k.stem for k in kernels]}; merge them into one kernel")
