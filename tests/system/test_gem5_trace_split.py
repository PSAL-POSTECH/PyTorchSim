"""gem5's coprocessor-trace/2 Perfetto file is split into one process per cycle-table row: a dump closes the
window the latest reset opened, slices are clipped, counters carry their held value in, ts is rebased to the
window, and a trace that disagrees with the table is refused. coprocessor_trace_two_tiles.pftrace is the
first two windows of the real sample (the file prefix up to its second dump); the sample is whole."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("TORCHSIM_DIR") or os.path.dirname(os.path.dirname(HERE)))

from pytorchsim.simulator import perfetto_trace  # noqa: E402
from pytorchsim.simulator.perfetto_trace import _bytes, _str, _vi  # noqa: E402
from pytorchsim.simulator.timing import gem5  # noqa: E402

DATA = os.path.join(HERE, "data", "viewer")
SAMPLE_CYCLES = [16, 284, 14, 282]
MXU = "TPUV6E_MXU.rows_in_flight"


def _read(name):
    with open(os.path.join(DATA, name), "rb") as fh:
        return fh.read()


def _refusal(data, cycles):
    try:
        gem5.split_trace_by_tile(data, cycles)
    except ValueError as e:
        return str(e)
    return "no refusal"


def _descriptor(uuid, name, parent=None, process=False, counter=False):
    body = _vi(1, uuid) + _str(2, name)
    body += _bytes(3, _vi(1, 1)) if process else b""
    body += _vi(5, parent) if parent else b""
    body += _bytes(8, b"") if counter else b""
    return _bytes(1, _vi(10, 1) + _bytes(60, body))


def _event(ts, kind, track, name=None, value=None, **args):
    body = _vi(9, kind) + _vi(11, track) + (_str(23, name) if name else b"")
    for k, v in args.items():
        body += _bytes(4, _str(10, k) + (_str(6, v) if isinstance(v, str) else _vi(4, v)))
    body += _vi(30, value) if value is not None else b""
    return _bytes(1, _vi(8, ts) + _vi(10, 1) + _bytes(11, body))


def _gem5_file(slices, counters, windows, schema="coprocessor-trace/2"):
    """A coprocessor trace as gem5 writes it: slices (begin, end, name), counter points (ts, value) and
    reset/dump windows, hand-encoded with the codec's wire helpers."""
    out = [_descriptor(1, "coprocessor m", process=True), _descriptor(2, "stats", 1),
           _event(0, 3, 2, "schema", schema=schema), _descriptor(16, "insn", 1),
           _descriptor(1000, "G.c", 1, counter=True)]
    for lo, hi in windows:
        out += [_event(lo, 3, 2, "reset", index=0), _event(hi, 3, 2, "dump", index=0)]
    for a, b, name in slices:
        out += [_event(a, 1, 16, name, id=1), _event(b, 2, 16)]
    out += [_event(ts, 4, 1000, value=v) for ts, v in counters]
    return b"".join(out)


def _integral(points, end):
    """sum of value * cycles held over [first point, end) for [(ts, value)]."""
    edges = points + [(end, 0)]
    return sum(v * (edges[i + 1][0] - t) for i, (t, v) in enumerate(points))


def test_each_window_becomes_the_row_of_its_tile():
    rows = gem5.split_trace_by_tile(_read("coprocessor_trace_two_tiles.pftrace"), [16, 284])
    assert [(r["row"], r["cycles"], len(r["insns"])) for r in rows] == [(0, 16, 1), (1, 284, 14)]
    for r in rows:
        assert all(0 <= ts and ts + dur <= r["cycles"] for _, _, ts, dur, _ in r["insns"]), r["row"]
        assert all(0 <= ts <= r["cycles"] for _, ts, _ in r["counters"]), r["row"]


def test_a_window_the_cycle_table_does_not_have_is_refused():
    data = _read("coprocessor_trace_two_tiles.pftrace")
    assert "2 tile windows, the cycle table 3 rows" in _refusal(data, [16, 284, 14])
    assert "window 1 is 284 cycles, the cycle table says 280" in _refusal(data, [16, 280])


def test_another_schema_is_refused():
    data = _read("coprocessor_trace_two_tiles.pftrace")
    assert b"coprocessor-trace/2" in data
    assert "coprocessor-trace/1" in _refusal(data.replace(b"coprocessor-trace/2", b"coprocessor-trace/1"),
                                             [16, 284])
    no_schema = _descriptor(1, "coprocessor m", process=True) + _descriptor(2, "stats", 1)
    assert "schema None" in _refusal(no_schema, [10])


def test_slices_are_clipped_and_counters_carry_their_held_value_into_each_window():
    data = _gem5_file(slices=[(5, 15, "a"), (18, 25, "b"), (40, 60, "c")],
                      counters=[(3, 2), (12, 5), (22, 0), (41, 7)], windows=[(10, 20), (30, 50)])
    r0, r1 = gem5.split_trace_by_tile(data, [10, 20])
    assert r0["insns"] == [("insn", "a", 0, 5, {"id": 1}), ("insn", "b", 8, 2, {"id": 1})]
    assert r1["insns"] == [("insn", "c", 10, 10, {"id": 1})]
    assert r0["counters"] == [("G.c", 0, 2), ("G.c", 2, 5)]
    assert r1["counters"] == [("G.c", 0, 0), ("G.c", 11, 7)]
    assert _integral([(t, v) for _, t, v in r0["counters"]], 10) == 2 * 2 + 5 * 8


def test_no_trace_file_writes_no_timeline_and_a_trace_writes_one():
    labels = list(zip(range(4), [0, 0, 1, 1]))
    table = [(c, 0) for c in SAMPLE_CYCLES]
    with tempfile.TemporaryDirectory() as m5out:
        path = os.path.join(m5out, "tog_tile_timeline.pftrace")
        assert gem5.write_tile_timeline(m5out, table, labels, path) is None
        assert not os.path.exists(path)
        with open(os.path.join(m5out, gem5.TRACE_FILE), "wb") as fh:
            fh.write(_read("coprocessor_trace_sample.pftrace"))
        assert gem5.write_tile_timeline(m5out, table, labels, path) == path
        trace = perfetto_trace.read_trace(path)
    assert trace["otherData"]["schema"] == perfetto_trace.TILE_TIMELINE_SCHEMA
    names = [e["args"]["name"] for e in trace["traceEvents"] if e["name"] == "process_name"]
    assert names == ["tile row 0: tile 0 call 0 (16 cycles)", "tile row 1: tile 1 call 0 (284 cycles)",
                     "tile row 2: tile 2 call 1 (14 cycles)",
                     "tile row 3: tile 3 call 1 (282 cycles)"], names
    slices = [e["pid"] for e in trace["traceEvents"] if e["ph"] == "X"]
    assert [slices.count(p) for p in (1, 2, 3, 4)] == [1, 14, 1, 14]


def test_the_real_sample_has_four_windows_and_tile_1_issues_11968_mxu_row_cycles():
    rows = gem5.split_trace_by_tile(_read("coprocessor_trace_sample.pftrace"), SAMPLE_CYCLES)
    assert [r["cycles"] for r in rows] == SAMPLE_CYCLES
    mxu = [(ts, v) for n, ts, v in rows[1]["counters"] if n == MXU]
    assert _integral(mxu, 284) == 11968
    labels = [(i, 0) for i in range(4)]
    trace = perfetto_trace.decode_trace(perfetto_trace.encode_trace(gem5.build_timeline_trace(rows, labels)))
    pts = sorted((round(e["ts"] * 1000), e["args"]["delta"]) for e in trace["traceEvents"]
                 if e["ph"] == "C" and e["name"] == MXU and e["pid"] == 2)
    assert _integral(pts, 284) == 11968


if __name__ == "__main__":
    test_each_window_becomes_the_row_of_its_tile()
    test_a_window_the_cycle_table_does_not_have_is_refused()
    test_another_schema_is_refused()
    test_slices_are_clipped_and_counters_carry_their_held_value_into_each_window()
    test_no_trace_file_writes_no_timeline_and_a_trace_writes_one()
    test_the_real_sample_has_four_windows_and_tile_1_issues_11968_mxu_row_cycles()
    print("PASS")
