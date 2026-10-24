"""TOGSim's results reach the frontend through the file it writes with --result, never its log. The
Simulator here is a stub that writes the result of a tpuv6e run of test_matmul.py, so no TOGSim is needed."""
import os
import stat
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import config
from pytorchsim.simulator.togsim import TOGSimProcess

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "togsim_tpuv6e.json")
STUB = """#!/bin/sh
while [ $# -gt 0 ]; do
  [ "$1" = --models_list ] && cat "$2" > /dev/null
  [ "$1" = --result ] && {write}
  shift
done
"""


def _checkout(root, writes=True):
    """A TOGSim checkout whose Simulator drains its command stream and, if `writes`, copies RESULT to
    its --result path; the machine file and a kernel directory beside it."""
    os.makedirs(os.path.join(root, "build", "bin"))
    simulator = config.togsim_simulator(root)
    with open(simulator, "w") as f:
        f.write(STUB.format(write=f'cp {RESULT} "$2"' if writes else "true"))
    os.chmod(simulator, os.stat(simulator).st_mode | stat.S_IXUSR)
    machine = os.path.join(root, "machine.yml")
    with open(machine, "w") as f:
        f.write("core_freq_mhz: 1\n")
    kernel = os.path.join(root, "k")
    os.makedirs(kernel)
    for name in ("tog.so", "tog_cycles.tsv", "tog_units.tsv"):
        open(os.path.join(kernel, name), "w").close()
    return machine, os.path.join(kernel, "tile_graph.onnx"), os.path.join(kernel, "attribute")


def _logs_in(root):
    return mock.patch.dict(os.environ, TORCHSIM_LOG_PATH=os.path.join(root, "logs"))


def test_standalone_reads_the_result_file():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root)
        path = TOGSimProcess.run_standalone(handle, attribute, autotune_mode=True, config_path=machine,
                                            togsim_path=root)
        got = TOGSimProcess.read_result(path)
        units = got["units"]["Core [0]"]
        assert got["total_cycles"] == got["core_cycles"]["0"] > 0
        assert {config.CONFIG_MACHINE.togsim_unit, "Vpu"} <= set(units)
        assert len(got["dram"]["channels"]) == 32 and got["dram"]["combined"]["reads"] > 0


def test_session_writes_its_result_file():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root)
        sim = TOGSimProcess(config_path=machine, togsim_path=root)
        sim.launch_kernel(0, 0, handle, attribute)
        sim.shutdown()
        assert TOGSimProcess.read_result(sim.result_path)["total_cycles"] > 0


def test_a_run_without_a_result_file_is_an_error():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root, writes=False)
        path = TOGSimProcess.run_standalone(handle, attribute, autotune_mode=True, config_path=machine,
                                            togsim_path=root)
        try:
            TOGSimProcess.read_result(path)
        except FileNotFoundError:
            return
        raise AssertionError(f"{path} was read though the Simulator wrote no result")


if __name__ == "__main__":
    test_standalone_reads_the_result_file()
    test_session_writes_its_result_file()
    test_a_run_without_a_result_file_is_an_error()
    print("PASS")
