"""get_result_from_file reads the statistics TOGSim prints today: one row per unit with the unit
table (the default), a row per array without it. The unit table is cut from a test_matmul run on
tpuv6e.yml (libtpuv6.so, array unit TPUV6_MXU), the per-array log from a 2048x2048x2048 run."""
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim.simulator.togsim import TOGSimulator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_unit_table_rows():
    core, channels, avg_bw, wall, total = TOGSimulator.get_result_from_file(
        os.path.join(DATA, "togsim_units.log"), "TPUV6_MXU")
    units = core["units"]["Core [0]"]
    assert units["TPUV6_MXU"] == {"utilization": 8.22, "active_cycles": 256, "idle_cycles": 2860}
    assert units["Vpu"]["active_cycles"] == 128
    assert core["Systolic_Array_Utilization"] == 8.22
    assert core["MatMul_active_cycle"] == 256
    assert core["Vector_Unit_Utilization"] == 4.11
    assert core["Total_cycle"] == total == 3116
    assert channels == {"CH[0]": 120.73, "CH[1]": 120.73} and avg_bw == 120.72
    assert wall == float("inf")


def test_per_array_rows():
    core, channels, avg_bw, _, total = TOGSimulator.get_result_from_file(
        os.path.join(DATA, "togsim_arrays.log"), "SYSTOLIC")
    units = core["units"]["Core [0]"]
    assert units["Systolic array [0]"]["active_cycles"] == 142475
    assert units["Vector unit"] == {"utilization": 2.39, "active_cycles": 5320, "idle_cycles": 0}
    assert core["Systolic_Array_Utilization"] == (64.06 + 0.0) / 2
    assert core["MatMul_active_cycle"] == 142475
    assert total == 222415 and avg_bw == 22.13 and len(channels) == 2


def test_unfinished_log_has_the_same_shape():
    with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
        log.write("[info] Core [0] : Total_cycles: 3\n")
        log.flush()
        assert TOGSimulator.get_result_from_file(log.name, "SYSTOLIC") == ({}, {}, 0.0, float("inf"), float("inf"))


if __name__ == "__main__":
    test_unit_table_rows()
    test_per_array_rows()
    test_unfinished_log_has_the_same_shape()
    print("PASS")
