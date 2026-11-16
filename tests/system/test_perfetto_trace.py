"""The Perfetto codec: wire framing, the packets encode_trace writes, the event dict decode_trace returns,
and decode_tracks over a trace with inline names."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("TORCHSIM_DIR") or os.path.dirname(os.path.dirname(HERE)))

from pytorchsim.simulator import perfetto_trace  # noqa: E402


def _canon(trace):
    """Events of a trace as sorted tuples with ts and dur rounded to ns and None args dropped."""
    out = []
    for e in trace["traceEvents"]:
        e = {k: v for k, v in e.items() if k != "cname"}
        for key in ("ts", "dur"):
            if key in e:
                e[key] = round(e[key] * 1000)
        out.append(json.dumps(e, sort_keys=True, default=str))
    return sorted(out)


def test_wire_codec_frames_varints_and_length_delimited_fields():
    """Varints are 7 bits a byte (300 -> ac 02, -1 -> ten bytes), a string is key, length, bytes, and a
    truncated message raises."""
    assert perfetto_trace._varint(300) == b"\xac\x02" and len(perfetto_trace._varint(-1)) == 10
    msg = perfetto_trace._vi(1, 150) + perfetto_trace._str(2, "hé") + perfetto_trace._f64(3, 0.5)
    assert msg[:3] == b"\x08\x96\x01"
    assert perfetto_trace.parse_message(msg) == [(1, 0, 150), (2, 2, "hé".encode()),
                                                 (3, 1, b"\x00\x00\x00\x00\x00\x00\xe0?")]
    for cut in (msg[:-3], msg[:6], b"\x08"):
        try:
            perfetto_trace.parse_message(cut)
        except ValueError:
            continue
        raise AssertionError(f"accepted {cut!r}")


def test_encoded_trace_is_packets_with_descriptors_and_track_events():
    """Trace.packet=1 repeated: the first packet clears incremental state and interns names, then track
    descriptors (process, thread, counter) and one TrackEvent packet per begin, end, instant and counter."""
    trace = {"traceEvents": [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "core"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 7, "args": {"name": "Kernels"}},
        {"ph": "X", "name": "k", "cat": "kernel", "pid": 1, "tid": 7, "ts": 2.0, "dur": 3.0,
         "args": {"n": 4}},
        {"ph": "C", "name": "util", "pid": 1, "ts": 2.0, "args": {"%": 9.5}}], "otherData": {"a": "b"}}
    packets = [dict((f, v) for f, _, v in perfetto_trace.parse_message(v))
               for f, _, v in perfetto_trace.parse_message(perfetto_trace.encode_trace(trace)) if f == 1]
    assert packets[0][13] == 1 and 12 in packets[0] and all(p[10] == 1 for p in packets)
    kinds = [perfetto_trace.parse_message(p[60]) for p in packets if 60 in p]
    names = [dict((f, v) for f, _, v in k) for k in kinds]
    assert any(3 in n for n in names) and any(4 in n for n in names) and any(8 in n for n in names)
    events = [(p[8], dict((f, v) for f, _, v in perfetto_trace.parse_message(p[11])))
              for p in packets if 11 in p]
    assert [(ts, e[9]) for ts, e in events if ts] == [(2000, 1), (2000, 4), (5000, 2)], events
    assert all(p[13] == 2 for p in packets if 11 in p)


def test_decode_returns_the_events_that_were_encoded():
    """Slices, instants (global, process and thread scope), counters (also two keys), names, sort indexes,
    args of every type and otherData survive the encode and decode round trip, ts to the ns."""
    trace = {"traceEvents": [
        {"ph": "M", "name": "process_name", "pid": 3, "tid": 0, "args": {"name": "P"}},
        {"ph": "M", "name": "process_sort_index", "pid": 3, "tid": 0, "args": {"sort_index": 2}},
        {"ph": "M", "name": "thread_name", "pid": 3, "tid": 1, "args": {"name": "T"}},
        {"ph": "M", "name": "thread_sort_index", "pid": 3, "tid": 1, "args": {"sort_index": 0}},
        {"ph": "X", "name": "outer", "cat": "c", "pid": 3, "tid": 1, "ts": 1.5, "dur": 10.0,
         "args": {"i": -5, "f": 0.25, "s": "ü", "b": True, "n": None, "d": {"x": [1, "y"]}, "e": [],
                  "o": {}}},
        {"ph": "X", "name": "inner", "pid": 3, "tid": 1, "ts": 1.5, "dur": 4.0, "args": {}},
        {"ph": "X", "name": "zero", "pid": 3, "tid": 1, "ts": 8.0, "dur": 0.0, "args": {}},
        {"ph": "i", "s": "g", "name": "run", "pid": 0, "tid": 0, "ts": 0.0, "args": {"log": "a"}},
        {"ph": "i", "s": "p", "name": "pi", "pid": 3, "tid": 0, "ts": 2.0, "args": {}},
        {"ph": "i", "s": "t", "name": "ti", "pid": 3, "tid": 1, "ts": 3.0, "args": {"k": 1}},
        {"ph": "C", "name": "gb", "pid": 3, "ts": 1.0, "args": {"GB/s": 2.5}},
        {"ph": "C", "name": "mxu", "pid": 3, "ts": 1.0, "args": {"rows": 4, "active": 8}}],
        "otherData": {"time_axis": "1 us = 5 cycles", "runs": ["a", "b"], "core_mhz": 1751}}
    back = perfetto_trace.decode_trace(perfetto_trace.encode_trace(trace))
    assert back["otherData"] == trace["otherData"]
    got, want = _canon(back), _canon(trace)
    assert [json.loads(g) for g in got] == [json.loads(w) for w in want], (got, want)


def test_decode_tracks_reads_inline_names_parents_and_counter_values():
    """gem5's form: a descriptor with a name and a parent, events with the name inline (field 23), string
    and int annotations named inline (field 10) and an integer counter_value (field 30)."""
    v, f, b = perfetto_trace._vi, perfetto_trace._str, perfetto_trace._bytes
    desc = [v(1, 1) + f(2, "proc") + b(3, v(1, 1)), v(1, 9) + f(2, "insn") + v(5, 1),
            v(1, 7) + f(2, "g.c") + v(5, 1) + b(8, b"")]
    ann = b(4, f(10, "id") + v(4, 3)) + b(4, f(10, "bits") + f(6, "0x1"))
    events = [(5, v(9, 1) + v(11, 9) + f(23, "add") + ann), (8, v(9, 2) + v(11, 9)),
              (6, v(9, 4) + v(11, 7) + v(30, 4))]
    data = b"".join(b(1, v(10, 1) + b(60, d)) for d in desc)
    data += b"".join(b(1, v(8, ts) + v(10, 1) + b(11, e)) for ts, e in events)
    tracks, got = perfetto_trace.decode_tracks(data)
    assert tracks[1]["process"]
    assert tracks[9] == {"name": "insn", "parent": 1, "process": False, "counter": False}
    assert tracks[7]["counter"]
    assert [(e["ts"], e["kind"], e["track"], e["name"], e["args"], e["value"]) for e in got] == [
        (5, 1, 9, "add", {"id": 3, "bits": "0x1"}, None), (8, 2, 9, None, {}, None), (6, 4, 7, None, {}, 4)]
