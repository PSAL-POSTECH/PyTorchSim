"""new_launch_dir: a repeated pid and count get new directories, workdir_of finds the workdir from one and
from its copy node's, remove_launch_dir removes this process's alone, and exit removes them unless
TORCHSIM_KEEP_LAUNCHES=1. Breaks on a launch-<pid>-<n> name alone, or launches piling up across runs."""
import itertools
import os
import subprocess
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim import contract  # noqa: E402


def test_a_repeated_pid_and_count_gets_new_directories():
    with tempfile.TemporaryDirectory() as workdir:
        earlier = os.path.join(workdir, f"{contract.LAUNCH_DIR_PREFIX}-{os.getpid()}-0")
        os.makedirs(earlier)
        with mock.patch.object(contract, "_LAUNCHES", itertools.count()), mock.patch.object(contract, "_MADE", {}):
            first = contract.new_launch_dir(workdir, (), {"kernel": "k"})
            second = contract.new_launch_dir(workdir, (), {"kernel": "k"})
            assert len({earlier, first, second}) == 3, (first, second)
            for handle in ("tile_graph.onnx", "copy_node_1/tile_graph.onnx"):
                assert contract.workdir_of(os.path.join(first, handle)) == workdir, handle
            contract.remove_launch_dir(workdir)
        assert os.path.isdir(earlier) and not os.path.exists(first) and not os.path.exists(second)


def _launch_in_child(workdir, keep):
    env = dict(os.environ, TORCHSIM_KEEP_LAUNCHES="1" if keep else "0")
    code = ("import sys; from pytorchsim import contract; "
            "print(contract.new_launch_dir(sys.argv[1], (), {'kernel': 'k'}))")
    return subprocess.run([sys.executable, "-c", code, workdir], env=env, check=True, capture_output=True,
                          text=True, cwd=os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim")).stdout.strip()


def test_launches_go_at_exit_unless_kept():
    with tempfile.TemporaryDirectory() as workdir:
        gone = _launch_in_child(workdir, keep=False)
        kept = _launch_in_child(workdir, keep=True)
        assert gone and not os.path.exists(gone), gone
        assert os.path.isfile(os.path.join(kept, contract.LAUNCH_RECORD)), kept


if __name__ == "__main__":
    test_a_repeated_pid_and_count_gets_new_directories()
    test_launches_go_at_exit_unless_kept()
    print("PASS")
