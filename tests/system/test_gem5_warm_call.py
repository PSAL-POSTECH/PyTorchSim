"""The cycle and unit tables take the cycle binary's last (warm) call, and only from a sample count that
is exactly the compiler's CYCLE_CALLS times the compute tiles. Stub gem5 output, no gem5 run."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim.driver import compiler_bridge
from pytorchsim.simulator import timing

KO = compiler_bridge.compiler_contract("kernel_object")


def _calls(n, fn):
    said = KO.CYCLE_CALLS
    KO.CYCLE_CALLS = n
    try:
        return fn()
    finally:
        KO.CYCLE_CALLS = said


def _refused(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    raise AssertionError("not refused")


def _m5out(d, ops):
    """A gem5 output directory with one dump per entry of ops (its committed micro-ops), the
    preamble before the first and the end-of-run dump after the last."""
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"system": {"cpu": {"executeIssueLimit": 8}}}, f)
    dump = ("\n---------- Begin Simulation Statistics ----------\n"
            "system.cpu.numCycles {n} # Number of cpu cycles simulated (Cycle)\n"
            "system.cpu.commitStats0.numOps {n} # Number of ops (Count)\n")
    with open(os.path.join(d, "stats.txt"), "w") as f:
        f.write("preamble\n" + "".join(dump.format(n=n) for n in ops) + dump.format(n=0))
    return d


def test_two_calls_of_two_tiles_take_the_second():
    assert _calls(2, lambda: timing.last_call([10, 11, 20, 21], 2, "x")) == [20, 21]
    with tempfile.TemporaryDirectory() as d:
        rows = _calls(2, lambda: timing.unit_rows(_m5out(d, [10, 11, 20, 21]), 2))
    assert rows == [(0, "Issue", "micro-ops", 8, 20), (1, "Issue", "micro-ops", 8, 21)]


def test_a_count_off_by_one_call_is_refused():
    said = _calls(2, lambda: _refused(lambda: timing.last_call([10, 11, 12], 1, "cycle sample(s)")))
    assert "3 cycle sample(s)" in said and "2 call(s) of 1 compute tile(s) make 2" in said
    with tempfile.TemporaryDirectory() as d:
        _calls(2, lambda: _refused(lambda: timing.unit_rows(_m5out(d, [10, 11, 12]), 1)))


def test_one_call_is_unchanged():
    assert _calls(1, lambda: timing.last_call([10, 11], 2, "x")) == [10, 11]
    assert "3 x" in _calls(1, lambda: _refused(lambda: timing.last_call([10, 11, 12], 2, "x")))


def test_the_contract_states_the_calls():
    assert isinstance(KO.CYCLE_CALLS, int) and KO.CYCLE_CALLS >= 1


if __name__ == "__main__":
    test_two_calls_of_two_tiles_take_the_second()
    test_a_count_off_by_one_call_is_refused()
    test_one_call_is_unchanged()
    test_the_contract_states_the_calls()
    print("PASS")
