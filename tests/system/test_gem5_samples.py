"""The timing tables hold a row per dump the compiler's one reader of m5out returns, in its order:
dumps out of program order still land on the row of their tile and call, each row's overlap is its
own tile's, and a run that measured a tile twice in one call is refused rather than tabled."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim.simulator.timing import gem5


def _dump(tile, call, cycles, admitted):
    return "\n".join([
        "---------- Begin Simulation Statistics ----------",
        f"system.cpu.numCycles {cycles} # Number of cpu cycles simulated (Cycle)",
        f"system.cpu.commitStats0.numOps {admitted} # Number of ops committed (Count)",
        f"system.cpu.units.Vpu.admitted {admitted} # cycles admitted at port busy (Count)",
        "system.cpu.units.Vpu.capacity 4 # capacity per cycle (Count)",
        f"system.sample.tileId {tile} # tile id the opening marker passed (Unspecified)",
        f"system.sample.call {call} # call the opening marker passed (Unspecified)",
        "---------- End Simulation Statistics   ----------", ""])


def _refusal(m5out, n_tiles):
    try:
        gem5.tile_tables(m5out, [None] * n_tiles)
    except ValueError as e:
        return str(e)
    return "no refusal"


def _m5out(keys):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"system": {"cpu": {"executeIssueLimit": 8}}}, f)
    with open(os.path.join(d, "stats.txt"), "w") as f:
        f.write("".join(_dump(*k) for k in keys) + _dump(-1, -1, 5, 1))
    return d


def test_every_call_on_its_row():
    m5out = _m5out([(1, 0, 900, 9), (0, 0, 800, 8), (1, 1, 300, 3), (0, 1, 200, 2)])
    table, rows = gem5.tile_tables(m5out, [None, 100])
    assert table == [(800, 0), (900, 800), (200, 0), (300, 200)]
    assert rows == [(0, "Issue", "micro-ops", 8, 8), (0, "Vpu", "cycles", 4, 8),
                    (1, "Issue", "micro-ops", 8, 9), (1, "Vpu", "cycles", 4, 9),
                    (2, "Issue", "micro-ops", 8, 2), (2, "Vpu", "cycles", 4, 2),
                    (3, "Issue", "micro-ops", 8, 3), (3, "Vpu", "cycles", 4, 3)]


def test_tile_measured_twice_is_refused():
    m5out = _m5out([(0, 0, 800, 8), (0, 0, 810, 8), (0, 1, 200, 2), (1, 1, 300, 3)])
    assert "two dumps name tile 0 call 0" in _refusal(m5out, 2)


def test_missing_tile_is_refused():
    m5out = _m5out([(0, 0, 800, 8), (0, 1, 200, 2)])
    assert "missing (tile, call) [(1, 0), (1, 1)]" in _refusal(m5out, 2)


if __name__ == "__main__":
    test_every_call_on_its_row()
    test_tile_measured_twice_is_refused()
    test_missing_tile_is_refused()
    print("PASS")
