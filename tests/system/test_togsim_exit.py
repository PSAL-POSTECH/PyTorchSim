"""A Simulator that exits non-zero is a RuntimeError on both paths: run_standalone carries its last
line, shutdown its status. The Simulator here is a stub, so no TOGSim is needed."""
import os
import stat
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import config
from pytorchsim.simulator.timing.togsim import TOGSimProcess

REFUSAL = "[error] [TOGSim] TOG k/tog.so: work-item 0 (key 10) needs 49153 bytes in the spad at once"
STUB = """#!/bin/sh
while [ $# -gt 0 ]; do [ "$1" = --models_list ] && cat "$2" > /dev/null; shift; done
echo "[info] [TOGSim] loading"
echo "{refusal}"
exit {status}
"""


def _checkout(root, status):
    """A TOGSim checkout whose Simulator drains its command stream, prints REFUSAL and exits `status`,
    and a machine file in it; the simulator is handed both explicitly."""
    os.makedirs(os.path.join(root, "build", "bin"))
    simulator = config.togsim_simulator(root)
    with open(simulator, "w") as f:
        f.write(STUB.format(refusal=REFUSAL, status=status))
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
    """TORCHSIM_LOG_PATH pointed into `root` for the block, and the environment put back after it."""
    return mock.patch.dict(os.environ, TORCHSIM_LOG_PATH=os.path.join(root, "logs"))


def _raised(fn):
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    return None


def test_standalone_refusal_is_a_runtime_error():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root, 1)
        said = _raised(lambda: TOGSimProcess.run_standalone(
            handle, attribute, autotune_mode=True, config_path=machine, togsim_path=root))
        assert said is not None and "status 1" in said and REFUSAL in said, said


def test_session_refusal_is_a_runtime_error():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root, 1)
        sim = TOGSimProcess(config_path=machine, togsim_path=root)
        sim.launch_kernel(0, 0, handle, attribute)
        said = _raised(sim.shutdown)
        assert said is not None and "status 1" in said, said
        assert sim.process is None and not os.path.exists(sim.fifo_dir)


def test_session_that_ran_shuts_down_quietly():
    with tempfile.TemporaryDirectory() as root, _logs_in(root):
        machine, handle, attribute = _checkout(root, 0)
        sim = TOGSimProcess(config_path=machine, togsim_path=root)
        sim.launch_kernel(0, 0, handle, attribute)
        assert _raised(sim.shutdown) is None


def test_environment_is_left_as_found():
    """No TOGSIM_CONFIG (which the compiler bridge refuses) or log path outlives a test here."""
    before = dict(os.environ)
    test_standalone_refusal_is_a_runtime_error()
    test_session_that_ran_shuts_down_quietly()
    assert dict(os.environ) == before, sorted(set(os.environ.items()) ^ set(before.items()))


if __name__ == "__main__":
    test_standalone_refusal_is_a_runtime_error()
    test_session_refusal_is_a_runtime_error()
    test_session_that_ran_shuts_down_quietly()
    test_environment_is_left_as_found()
    print("PASS")
