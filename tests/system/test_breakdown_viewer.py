"""The perf viewer reads the breakdown wallclock writes: its file under the dump path, the compile
records under the key the writer uses, and torchsim-compile first among the groups; a breakdown that
cannot be written is logged; a log's compile clock is its workdir's, above its launch directory."""
import json
import os
import runpy
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

from pytorchsim import contract  # noqa: E402
from pytorchsim.simulator import wallclock  # noqa: E402

VIEWER = runpy.run_path(os.path.join(TORCHSIM_DIR, "scripts", "util_viewer.py"))
KERNEL = "triton_poi_fused_add_0"
TIMING = {"label": KERNEL, "stages": [{"name": "transform", "seconds": 0.5},
                                      {"name": "lower", "seconds": 0.25}]}


def _written_breakdown(root):
    """The dump path a run with TORCHSIM_BREAKDOWN=1 leaves: one compiled kernel, one Spike launch."""
    workdir = os.path.join(root, KERNEL)
    os.makedirs(workdir)
    with open(os.path.join(workdir, "timing.json"), "w") as f:
        json.dump(TIMING, f)
    env = {"TORCHSIM_BREAKDOWN": "1", "TORCHSIM_DUMP_PATH": root}
    with mock.patch.dict(os.environ, env), mock.patch.object(wallclock, "REC", wallclock.Record()):
        with wallclock.span(wallclock.TORCHSIM_COMPILE, KERNEL):
            wallclock.ingest_compile(workdir, KERNEL)
        with wallclock.span(wallclock.SPIKE, KERNEL):
            pass
        wallclock._at_exit()


def test_viewer_finds_the_file_wallclock_writes():
    """wallclock names its file breakdown_<run>.json; a viewer scanning for breakdown.json finds none."""
    with tempfile.TemporaryDirectory() as root:
        _written_breakdown(root)
        found = VIEWER["find_breakdowns"]([root])
        assert len(found) == 1, f"the viewer found {found} under {os.listdir(root)}"


def test_viewer_reads_the_compile_records():
    """Given the file, the compile records are read under the key the writer uses, and the group
    order is the writer's."""
    with tempfile.TemporaryDirectory() as root:
        _written_breakdown(root)
        found = VIEWER["find_breakdowns"]([os.path.join(root, f) for f in os.listdir(root)
                                           if f.endswith(".json")])
        assert VIEWER["timing_from_breakdowns"](found) == {KERNEL: TIMING}
        names = [g["name"] for g in VIEWER["breakdown_groups"](found[0])]
        assert names[:2] == [wallclock.TORCHSIM_COMPILE, wallclock.SPIKE], names


def test_a_breakdown_that_cannot_be_written_is_logged():
    with tempfile.TemporaryDirectory() as root, \
            mock.patch.dict(os.environ, {"TORCHSIM_BREAKDOWN": "1", "TORCHSIM_DUMP_PATH": root}), \
            mock.patch.object(wallclock, "REC", wallclock.Record()), \
            mock.patch.object(wallclock.logger, "warning") as warned:
        wallclock.REC.add(wallclock.SPIKE, KERNEL, 0.1)
        with mock.patch("builtins.open", side_effect=OSError("read-only")):
            wallclock._at_exit()
    assert warned.called and "could not write" in warned.call_args[0][0], warned.call_args


def test_a_launch_log_joins_its_workdir_timing():
    """TOGSim is handed the launch directory's tog.so: the timing.json joined is the workdir's above it."""
    with tempfile.TemporaryDirectory() as root:
        workdir = os.path.join(root, KERNEL)
        launch = contract.dir_for(workdir)
        with open(os.path.join(workdir, "timing.json"), "w") as f:
            json.dump(TIMING, f)
        timing = {}
        VIEWER["join_workdir_timing"]([{"tog_so": os.path.join(launch, "tog.so")}], timing)
        assert list(timing) == [workdir] and timing[workdir]["label"] == KERNEL, timing

if __name__ == "__main__":
    test_viewer_finds_the_file_wallclock_writes()
    test_viewer_reads_the_compile_records()
    test_a_breakdown_that_cannot_be_written_is_logged()
    test_a_launch_log_joins_its_workdir_timing()
    print("PASS")
