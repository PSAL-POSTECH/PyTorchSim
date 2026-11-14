"""gem5's coprocessor trace is split into one row per cycle-table row: a dump closes the window the
latest reset opened, events are rebased to it, and a trace that disagrees with the table is refused."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("TORCHSIM_DIR") or os.path.dirname(os.path.dirname(HERE)))

from pytorchsim.simulator.timing import gem5  # noqa: E402

TRACE = os.path.join(HERE, "data", "viewer", "coprocessor_trace_two_tiles.json")


def _trace():
    with open(TRACE) as fh:
        return json.load(fh)


def _refusal(trace, cycles):
    try:
        gem5.split_trace_by_tile(trace, cycles)
    except ValueError as e:
        return str(e)
    return "no refusal"


def test_each_window_becomes_the_row_of_its_tile():
    rows = gem5.split_trace_by_tile(_trace(), [16, 284])
    assert [(r["row"], r["cycles"]) for r in rows] == [(0, 16), (1, 284)]
    for r in rows:
        assert r["events"] and all(0 <= e["ts"] <= r["cycles"] for e in r["events"]), r["row"]
        assert all(e["ph"] in ("X", "C") for e in r["events"])
    assert all(e["ts"] + e.get("dur", 0) <= 284 for e in rows[1]["events"] if e["ph"] == "X")


def test_a_window_the_cycle_table_does_not_have_is_refused():
    assert "2 tile windows, the cycle table 3 rows" in _refusal(_trace(), [16, 284, 14])
    assert "window 1 is 284 cycles, the cycle table says 280" in _refusal(_trace(), [16, 280])


def test_another_schema_is_refused():
    t = _trace()
    t["otherData"]["schema"] = "coprocessor-trace/2"
    assert "coprocessor-trace/1" in _refusal(t, [16, 284])
    t = _trace()
    t["otherData"]["ts_unit"] = "ns"
    assert "in cycles" in _refusal(t, [16, 284])


def test_no_trace_file_writes_no_timeline_and_a_trace_writes_one():
    with tempfile.TemporaryDirectory() as m5out:
        path = os.path.join(m5out, "tog_tile_timeline.json")
        assert gem5.write_tile_timeline(m5out, [(16, 0), (284, 0)], path) is None
        assert not os.path.exists(path)
        with open(os.path.join(m5out, gem5.TRACE_FILE), "w") as fh:
            json.dump(_trace(), fh)
        assert gem5.write_tile_timeline(m5out, [(16, 0), (284, 0)], path) == path
        with open(path) as fh:
            doc = json.load(fh)
        assert doc["schema"] == gem5.TIMELINE_SCHEMA and [r["cycles"] for r in doc["rows"]] == [16, 284]


if __name__ == "__main__":
    test_each_window_becomes_the_row_of_its_tile()
    test_a_window_the_cycle_table_does_not_have_is_refused()
    test_another_schema_is_refused()
    test_no_trace_file_writes_no_timeline_and_a_trace_writes_one()
    print("PASS")
