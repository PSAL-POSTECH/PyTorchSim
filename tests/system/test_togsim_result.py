"""get_result_from_file reads the statistics TOGSim prints today: one row per unit with the unit
table (the default), a row per array without it. Both logs are cut from real 2048x2048x2048 runs."""
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim.simulator.togsim import TOGSimulator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_unit_table_rows():
    core, channels, avg_bw, wall, total = TOGSimulator.get_result_from_file(
        os.path.join(DATA, "togsim_units.log"))
    units = core["units"]["Core [0]"]
    assert units["Systolic"] == {"utilization": 50.66, "active_cycles": 65536, "idle_cycles": 63821}
    assert units["Vpu"]["active_cycles"] == 34132
    assert core["Systolic_Array_Utilization"] == 50.66
    assert core["MatMul_active_cycle"] == 65536
    assert core["Vector_Unit_Utilization"] == 26.39
    assert core["Total_cycle"] == total == 129357
    assert channels == {"CH[0]": 38.06, "CH[1]": 38.06} and avg_bw == 38.06
    assert wall == float("inf")


def test_per_array_rows():
    core, channels, avg_bw, _, total = TOGSimulator.get_result_from_file(
        os.path.join(DATA, "togsim_arrays.log"))
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
        assert TOGSimulator.get_result_from_file(log.name) == ({}, {}, 0.0, float("inf"), float("inf"))


if __name__ == "__main__":
    test_unit_table_rows()
    test_per_array_rows()
    test_unfinished_log_has_the_same_shape()
    print("PASS")
