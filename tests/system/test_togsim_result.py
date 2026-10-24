"""get_result_from_file reads the statistics TOGSim prints with the unit table: one row per unit, the
array named by the machine's systolic_array_unit. The log is cut from a tpuv6e run of test_matmul.py."""
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import config
from pytorchsim.simulator.timing.togsim import TOGSimProcess

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_unit_table_rows():
    core, channels, avg_bw, wall, total = TOGSimProcess.get_result_from_file(
        os.path.join(DATA, "togsim_tpuv6e.log"))
    units = core["units"]["Core [0]"]
    array = units[config.CONFIG_MACHINE.togsim_unit]
    assert array == {"utilization": 1.86, "active_cycles": 16, "idle_cycles": 846}
    assert core["Systolic_Array_Utilization"] == array["utilization"]
    assert core["MatMul_active_cycle"] == array["active_cycles"]
    assert core["Vector_Unit_Utilization"] == units["Vpu"]["utilization"] == 0.81
    assert core["Vector_active_cycle"] == 7
    assert core["Total_cycle"] == total == 862
    assert len(channels) == 32 and channels["CH[0]"] == 2.79 and avg_bw == 2.79
    assert wall == float("inf")


def test_unfinished_log_has_the_same_shape():
    with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
        log.write("[info] Core [0] : Total_cycles: 3\n")
        log.flush()
        assert TOGSimProcess.get_result_from_file(log.name) == ({}, {}, 0.0, float("inf"), float("inf"))


if __name__ == "__main__":
    test_unit_table_rows()
    test_unfinished_log_has_the_same_shape()
    print("PASS")
