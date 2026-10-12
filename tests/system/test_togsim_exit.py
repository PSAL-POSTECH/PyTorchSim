"""A Simulator that exits non-zero is a RuntimeError: run_standalone's carries the exit status and
the last line the Simulator printed. The Simulator here is a stub, so no TOGSim is needed."""
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import config
from pytorchsim.simulator.togsim import TOGSimulator

REFUSAL = "[error] [TOGSim] TOG k/tog.so: work-item 0 (key 10) needs 49153 bytes in the spad at once"
STUB = """#!/bin/sh
while [ $# -gt 0 ]; do [ "$1" = --models_list ] && cat "$2" > /dev/null; shift; done
echo "[info] [TOGSim] loading"
echo "{refusal}"
exit {status}
"""


def _checkout(root, status):
    """A TOGSim checkout whose Simulator drains its command stream, prints REFUSAL and exits `status`;
    the process's machine config and log directory are pointed into it."""
    os.makedirs(os.path.join(root, "build", "bin"))
    simulator = config.togsim_simulator(root)
    with open(simulator, "w") as f:
        f.write(STUB.format(refusal=REFUSAL, status=status))
    os.chmod(simulator, os.stat(simulator).st_mode | stat.S_IXUSR)
    machine = os.path.join(root, "machine.yml")
    with open(machine, "w") as f:
        f.write("core_freq_mhz: 1\n")
    os.environ["TOGSIM_CONFIG"] = machine
    os.environ["TORCHSIM_LOG_PATH"] = os.path.join(root, "logs")
    kernel = os.path.join(root, "k")
    os.makedirs(kernel)
    for name in ("tog.so", "tog_cycles.tsv", "tog_units.tsv"):
        open(os.path.join(kernel, name), "w").close()
    return machine, os.path.join(kernel, "tile_graph.onnx"), os.path.join(kernel, "attribute")


def _raised(fn):
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    return None


def test_standalone_refusal_is_a_runtime_error():
    with tempfile.TemporaryDirectory() as root:
        machine, handle, attribute = _checkout(root, 1)
        said = _raised(lambda: TOGSimulator.run_standalone(
            handle, attribute, autotune_mode=True, config_path=machine, togsim_path=root))
        assert said is not None and "status 1" in said and REFUSAL in said, said


if __name__ == "__main__":
    test_standalone_refusal_is_a_runtime_error()
    print("PASS")
