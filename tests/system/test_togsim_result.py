"""TOGSim's results reach the frontend through the file it writes with --result, never its log; a
session keeps the Simulator's output in its .log, a launch it queues has no result yet, and the attribute
copies its launches held go when it shuts down. The Simulator is a stub, so no TOGSim is needed."""
import os
import stat
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import config
from pytorchsim.backend import codecache
from pytorchsim.simulator import timing
from pytorchsim.simulator.timing.togsim import TOGSimProcess

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "togsim_tpuv6e.json")
SAID = "[info] [TOGSim] the stub ran"
STUB = """#!/bin/sh
echo "{said}"
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
        f.write(STUB.format(said=SAID, write=f'cp {RESULT} "$2"' if writes else "true"))
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
        with open(sim.log_path) as f:
            assert SAID in f.read(), sim.log_path
        stem = os.path.splitext(sim.result_path)[0]
        assert sim.log_path == stem + ".log" and os.path.isfile(stem + ".trace")


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


def test_a_session_removes_the_attribute_copies_its_launches_held():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root)
        os.makedirs(attribute)
        open(os.path.join(attribute, "arg0"), "w").close()
        sim = TOGSimProcess(config_path=machine, togsim_path=root)
        held = timing._held_for_the_session(sim, os.path.dirname(handle), attribute)
        assert os.path.isfile(os.path.join(held, "arg0"))
        sim.launch_kernel(0, 0, handle, held)
        sim.shutdown()
        assert not os.path.exists(held) and os.path.isdir(attribute)


def test_a_launch_queued_on_a_session_returns_no_result():
    launcher = codecache.TritonNPULauncher("k", "/nonexistent", {"kernel_name": "k"})
    with mock.patch.dict(os.environ, TORCHSIM_TIMING_MODE="1", TORCHSIM_FUNCTIONAL_MODE="0"), \
         mock.patch.object(codecache.timing, "run", return_value=None):
        assert launcher() is None


if __name__ == "__main__":
    test_standalone_reads_the_result_file()
    test_session_writes_its_result_file()
    test_a_run_without_a_result_file_is_an_error()
    test_a_session_removes_the_attribute_copies_its_launches_held()
    test_a_launch_queued_on_a_session_returns_no_result()
    print("PASS")
