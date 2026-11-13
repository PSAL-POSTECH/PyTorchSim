"""util_viewer reads every per-unit line and kernel of a TOGSim log, leaves a run that scheduled
nothing off the axis, and refuses to write a page with mojibake in it."""
import os
import re
import runpy
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

HERE = os.path.dirname(os.path.abspath(__file__))
TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TORCHSIM_DIR)

VIEWER = runpy.run_path(os.path.join(TORCHSIM_DIR, "scripts", "util_viewer.py"))
sys.path.insert(0, os.path.join(TORCHSIM_DIR, "scripts"))
import trace_timeline  # noqa: E402

LOG = os.path.join(HERE, "data", "viewer", "togsim_small.log")


def _quiet_main(argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = VIEWER["main"](argv)
    return rc, err.getvalue()


def test_every_unit_line_is_read_and_named():
    """The units of one log are whatever names the Core stat lines print, not a fixed list."""
    run = VIEWER["parse_log"](LOG)
    want = {"Issue", "TPUV6E_MISC.instructions", "TPUV6E_MXU.rows", "TPUV6E_SFU.instructions",
            "TPUV6E_XLU.elements", "Vpu"}
    assert set(run["cores"][0]["units"]) == want, run["cores"][0]["units"]
    assert set(run["wins"][0]["units"]) == want, run["wins"][0]["units"]
    assert run["cores"][0]["units"]["Vpu"] == 2 and run["cores"][0]["cycles"] == 1822
    assert run["config"]["systolic_array_unit"] == "TPUV6E_MXU" and run["cycles"] == 1822


def test_a_kernel_is_named_by_its_workdir_and_ends_at_its_cycle():
    run = VIEWER["parse_log"](LOG)
    assert [(k["id"], k["end"]) for k in run["kernels"]] == [(0, 1049)], run["kernels"]
    assert VIEWER["kernel_name"](run["kernels"][0]["tog"]) == ("triton_fixture", "dump")


def test_a_log_that_scheduled_nothing_is_an_empty_run():
    """A run with no kernel is parsed but left off the axis and listed as empty; a log TOGSim never
    finished is no run."""
    with tempfile.TemporaryDirectory() as root:
        with open(LOG) as fh:
            text = fh.read()
        idle = re.sub(r"^.*Scheduler.*\n", "", text, flags=re.M)
        path = os.path.join(root, "idle.log")
        with open(path, "w") as fh:
            fh.write(idle)
        unfinished = os.path.join(root, "unfinished.log")
        with open(unfinished, "w") as fh:
            fh.write(text.split("Total execution cycles")[0])
        runs = VIEWER["collect"]([root])
        assert [r["log"] for r in runs] == ["idle.log"], [r["log"] for r in runs]
        payload = VIEWER["build_payload"](runs)
        empty = [e["log"] for e in payload["empty"]]
        assert payload["runs"] == [] and empty == ["idle.log"], empty


def test_a_logged_container_path_is_read_from_the_host_prefix_it_is_given():
    path_map = VIEWER["_PATH_MAP"]
    assert not path_map, "no host path is built in"
    with tempfile.TemporaryDirectory() as root:
        path_map[:] = [("/data/dump/", root + "/")]
        try:
            run = VIEWER["parse_log"](LOG)
        finally:
            path_map.clear()
    assert run["kernels"][0]["tog"] == f"{root}/triton_fixture/launch-1-1-abcd/tog.so", run["kernels"]


def test_a_page_with_mojibake_is_refused():
    """Text that was UTF-8 read as Latin-1 upstream is an error naming the input, not a page."""
    with tempfile.TemporaryDirectory() as root:
        with open(LOG, encoding="utf-8") as fh:
            text = fh.read()
        path = os.path.join(root, "garbled.log")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text.replace("triton_fixture", "triton_caf\u00c3\u00a9"))
        VIEWER["_MOJIBAKE_INPUTS"].clear()
        out = os.path.join(root, "page.html")
        rc, err = _quiet_main([path, "-o", out])
        assert rc == 1 and "mojibake" in err and "garbled.log" in err, (rc, err)
        VIEWER["_MOJIBAKE_INPUTS"].clear()
        rc, err = _quiet_main([LOG, "-o", out])
        assert rc == 0, err
        with open(out, encoding="utf-8") as fh:
            assert "<html" in fh.read(2000).lower()


def test_a_compute_type_beyond_vector_and_matmul_has_its_own_lane():
    """COMP types other than vector (0) and matmul or preload (1, 2) are not drawn on a systolic lane."""
    lines = []
    for iid, (ct, issued) in enumerate(((0, 10), (3, 20))):
        for tag, cycle in (("INST_ISSUED", issued), ("INST_FINISHED", issued + 8)):
            detail = f"compute_type={ct} compute_cycle=8 sa=-1 tile=0"
            lines.append(f"[{cycle}][Core 0][{tag}][INST_ID={iid}] COMP ({detail})")
    events = trace_timeline.to_chrome(trace_timeline.parse(lines), 2)["traceEvents"]
    lane = {e["tid"] for e in events if e["ph"] == "X"}
    assert lane == {"vector", "unit3"}, lane


if __name__ == "__main__":
    test_every_unit_line_is_read_and_named()
    test_a_kernel_is_named_by_its_workdir_and_ends_at_its_cycle()
    test_a_log_that_scheduled_nothing_is_an_empty_run()
    test_a_logged_container_path_is_read_from_the_host_prefix_it_is_given()
    test_a_page_with_mojibake_is_refused()
    test_a_compute_type_beyond_vector_and_matmul_has_its_own_lane()
    print("PASS")
