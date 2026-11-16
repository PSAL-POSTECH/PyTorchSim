"""Chrome-style event dicts <-> Perfetto binary traces over a hand-written protobuf codec (no dependency).
Fields: Trace.packet=1; TracePacket ts=8 seq=10 track_event=11 interned=12 flags=13 track_descriptor=60;
TrackEvent categories=3 annotations=4 type=9 name_iid=10 track_uuid=11 double_counter=44 (more: commit)."""
import json
import math
import struct

SLICE_BEGIN, SLICE_END, INSTANT, COUNTER = 1, 2, 3, 4
SEQ_INCREMENTAL_STATE_CLEARED, SEQ_NEEDS_INCREMENTAL_STATE = 1, 2
CHILD_ORDERING_EXPLICIT = 3
SEQUENCE_ID = 1
GLOBAL_TRACK_NAME = "global"
METADATA_TRACK_NAME = "trace metadata"
METADATA_EVENT = "otherData"
TILE_TIMELINE_SCHEMA = "tog-tile-timeline/2"
_VARINT, _FIXED64, _LEN, _FIXED32 = 0, 1, 2, 5
_MASK = (1 << 64) - 1


def _varint(n):
    n &= _MASK
    out = bytearray()
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _vi(field, n):
    return _varint(field << 3 | _VARINT) + _varint(n)


def _bytes(field, payload):
    return _varint(field << 3 | _LEN) + _varint(len(payload)) + payload


def _str(field, text):
    return _bytes(field, text.encode("utf-8"))


def _f64(field, x):
    return _varint(field << 3 | _FIXED64) + struct.pack("<d", x)


def parse_message(buf):
    """The fields of one protobuf message as [(field number, wire type, value)]: an int for a varint, bytes
    for a length-delimited field, the 8 or 4 raw bytes for a fixed one. Truncated input raises ValueError."""
    out, i = [], 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == _VARINT:
            val, i = _read_varint(buf, i)
        elif wire == _LEN:
            size, i = _read_varint(buf, i)
            val, i = buf[i:i + size], i + size
            if len(val) != size:
                raise ValueError("truncated length-delimited field")
        elif wire in (_FIXED64, _FIXED32):
            size = 8 if wire == _FIXED64 else 4
            val, i = buf[i:i + size], i + size
            if len(val) != size:
                raise ValueError("truncated fixed field")
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((field, wire, val))
    return out


def _read_varint(buf, i):
    shift = val = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


class _Interner:
    """String -> iid, ids from 1, in first-use order."""

    def __init__(self):
        self.ids = {}

    def get(self, text):
        return self.ids.setdefault(text, len(self.ids) + 1)

    def entries(self, field):
        return b"".join(_bytes(field, _vi(1, i) + _str(2, t)) for t, i in self.ids.items())


def _encode_annotation(names, name, value):
    """A DebugAnnotation (name iid, then the value's field); empty containers go as JSON strings."""
    head = b"" if name is None else _vi(1, names.get(name))
    if value is None:
        return head
    if isinstance(value, bool):
        return head + _vi(2, int(value))
    if isinstance(value, int):
        return head + _vi(4, value)
    if isinstance(value, float):
        return head + _f64(5, value)
    if isinstance(value, str):
        return head + _str(6, value)
    if not value:
        return head + _str(9, json.dumps(value))
    if isinstance(value, dict):
        return head + b"".join(_bytes(11, _encode_annotation(names, k, v)) for k, v in value.items())
    return head + b"".join(_bytes(12, _encode_annotation(names, None, v)) for v in value)


class _Packets:
    """The TracePackets of one trace: descriptors first, then the events by time."""

    def __init__(self, trace):
        self.trace, self.cats, self.names, self.keys = trace, _Interner(), _Interner(), _Interner()
        self.next_uuid, self.pname, self.psort, self.tname, self.tsort = 0, {}, {}, {}, {}
        self.proc, self.thread, self.counter, self.global_uuid = {}, {}, {}, None
        self.descriptors, self.events = [], []

    def uuid(self):
        self.next_uuid += 1
        return self.next_uuid

    def build(self):
        tables = {"process_name": (self.pname, False), "process_sort_index": (self.psort, False),
                  "thread_name": (self.tname, True), "thread_sort_index": (self.tsort, True)}
        for e in self.trace["traceEvents"]:
            if e["ph"] == "M":
                table, per_thread = tables[e["name"]]
                table[(e["pid"], e["tid"]) if per_thread else e["pid"]] = next(iter(e["args"].values()))
        self._add_global_tracks()
        for pid in sorted({*self.pname, *self.psort}):
            self._get_process(pid)
        for pid, tid in sorted({*self.tname, *self.tsort}):
            self._get_thread(pid, tid)
        order = 0
        for e in self.trace["traceEvents"]:
            ph = e["ph"]
            if ph == "X":
                order = self._add_slice(e, order)
            elif ph == "i":
                self._add_event(INSTANT, self._get_instant_track(e), e["ts"], e.get("name"), e, order)
                order += 1
            elif ph == "C":
                order = self._add_counter(e, order)
        return self._finish()

    def _add_global_tracks(self):
        self.global_uuid, meta_uuid = self.uuid(), self.uuid()
        other = self.trace.get("otherData") or {}
        self.descriptors.append(_vi(1, self.global_uuid) + _str(2, GLOBAL_TRACK_NAME))
        desc = _vi(1, meta_uuid) + _str(2, METADATA_TRACK_NAME)
        if other.get("time_axis"):
            desc += _str(14, other["time_axis"])
        self.descriptors.append(desc)
        body = self._track_event(INSTANT, meta_uuid, METADATA_EVENT, None, other)
        self.events.append(((0, 1, 0, -1), body, 0))

    def _get_process(self, pid):
        if pid not in self.proc:
            uid = self.proc[pid] = self.uuid()
            body = _vi(1, pid)
            if pid in self.pname:
                body += _str(6, self.pname[pid])
            if pid in self.psort:
                body += _vi(3, self.psort[pid])
            self.descriptors.append(_vi(1, uid) + _bytes(3, body) + _vi(11, CHILD_ORDERING_EXPLICIT))
        return self.proc[pid]

    def _get_thread(self, pid, tid):
        if (pid, tid) not in self.thread:
            self._get_process(pid)
            uid = self.thread[(pid, tid)] = self.uuid()
            body = _vi(1, pid) + _vi(2, tid)
            if (pid, tid) in self.tname:
                body += _str(5, self.tname[(pid, tid)])
            rank = b""
            if (pid, tid) in self.tsort:
                body += _vi(3, self.tsort[(pid, tid)])
                rank = _vi(12, self.tsort[(pid, tid)] + 1)
            self.descriptors.append(_vi(1, uid) + _bytes(4, body) + rank)
        return self.thread[(pid, tid)]

    def _get_counter(self, pid, name, key, multi):
        if (pid, name, key) not in self.counter:
            parent = self._get_process(pid)
            uid = self.counter[(pid, name, key)] = self.uuid()
            self.descriptors.append(_vi(1, uid) + _str(2, f"{name} {key}" if multi else name) + _vi(5, parent)
                                    + _bytes(8, _str(6, key)) + _str(14, name))
        return self.counter[(pid, name, key)]

    def _get_instant_track(self, e):
        scope = e.get("s", "t")
        if scope == "g":
            return self.global_uuid
        if scope == "p":
            return self._get_process(e["pid"])
        return self._get_thread(e["pid"], e["tid"])

    def _track_event(self, kind, track, name, cat, args, counter=None):
        body = _vi(9, kind) + _vi(11, track)
        if cat:
            body += _vi(3, self.cats.get(cat))
        if name is not None:
            body += _vi(10, self.names.get(name))
        for k, v in (args or {}).items():
            body += _bytes(4, _encode_annotation(self.keys, k, v))
        if counter is not None:
            body += _f64(44, float(counter))
        return body

    def _add_event(self, kind, track, ts, name, e, order, counter=None, key=None):
        key = key or (self._ns(ts), 1, 0, order)
        self.events.append((key, self._track_event(kind, track, name, e.get("cat"), e.get("args"), counter),
                            self._ns(ts)))

    @staticmethod
    def _ns(us):
        return round(us * 1000)

    def _add_slice(self, e, order):
        track = self._get_thread(e["pid"], e["tid"])
        start = self._ns(e["ts"])
        dur = self._ns(e["dur"])
        begin = self._track_event(SLICE_BEGIN, track, e["name"], e.get("cat"), e.get("args"))
        self.events.append(((start, 1, -dur, order), begin, start))
        end_key = (start, 1, 0, order + 0.5) if dur == 0 else (start + dur, 0, -order, 0)
        self.events.append((end_key, _vi(9, SLICE_END) + _vi(11, track), start + dur))
        return order + 1

    def _add_counter(self, e, order):
        args = e["args"]
        for key, value in args.items():
            track = self._get_counter(e["pid"], e["name"], key, len(args) > 1)
            self._add_event(COUNTER, track, e["ts"], None, {}, order, counter=value)
            order += 1
        return order

    def _finish(self):
        interned = (self.cats.entries(1) + self.names.entries(2) + self.keys.entries(3))
        first = _vi(10, SEQUENCE_ID) + _bytes(12, interned) + _vi(13, SEQ_INCREMENTAL_STATE_CLEARED)
        out = [_bytes(1, first)]
        for desc in self.descriptors:
            out.append(_bytes(1, _vi(10, SEQUENCE_ID) + _bytes(60, desc)))
        for key, body, ts in sorted(self.events, key=lambda x: x[0]):
            out.append(_bytes(1, _vi(8, ts) + _vi(10, SEQUENCE_ID) + _bytes(11, body)
                              + _vi(13, SEQ_NEEDS_INCREMENTAL_STATE)))
        return b"".join(out)


def encode_trace(trace):
    """The binary Perfetto trace of a Chrome-style event dict {"traceEvents": [...], "otherData": {...}}.
    X becomes a slice begin and end on its thread track, i an instant, C one counter track per arg key and
    M the process and thread names and order; ts and dur are microseconds, stored as integer ns. otherData
    rides on an instant of the "trace metadata" track. Slices that overlap without nesting are not
    representable."""
    return _Packets(trace).build()


def write_trace(path, trace):
    """`trace` encoded and written to `path`; returns the byte count."""
    data = encode_trace(trace)
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


def _decode_annotation(names, buf):
    """(name, value) of one DebugAnnotation message."""
    name, value, entries, items, seen = None, None, [], [], None
    for field, _, val in parse_message(buf):
        if field == 1:
            name = names[val]
        elif field == 10:
            name = val.decode("utf-8")
        elif field in (2, 3, 4):
            value = bool(val) if field == 2 else (val - (1 << 64) if field == 4 and val >> 63 else val)
        elif field == 5:
            value = struct.unpack("<d", val)[0]
        elif field == 6:
            value = val.decode("utf-8")
        elif field == 9:
            value = json.loads(val.decode("utf-8"))
        elif field == 11:
            entries.append(_decode_annotation(names, val))
            seen = "dict"
        elif field == 12:
            items.append(_decode_annotation(names, val)[1])
            seen = "list"
    if seen == "dict":
        value = dict(entries)
    elif seen == "list":
        value = items
    return name, value


class _Decoder:
    def __init__(self):
        self.cats, self.names, self.keys, self.tracks, self.raw = {}, {}, {}, {}, []

    def read(self, data):
        for field, _, val in parse_message(data):
            if field != 1:
                continue
            fields = {}
            for f, _, v in parse_message(val):
                fields.setdefault(f, []).append(v)
            if fields.get(13, [0])[0] & SEQ_INCREMENTAL_STATE_CLEARED:
                self.cats, self.names, self.keys = {}, {}, {}
            for blob in fields.get(12, []):
                self._read_interned(blob)
            for blob in fields.get(60, []):
                self._read_track(blob)
            for blob in fields.get(11, []):
                self.raw.append((fields.get(8, [0])[0], blob))

    def _read_interned(self, blob):
        for field, _, val in parse_message(blob):
            table = {1: self.cats, 2: self.names, 3: self.keys}.get(field)
            if table is not None:
                f = {k: v for k, _, v in parse_message(val)}
                table[f[1]] = f[2].decode("utf-8")

    def _read_track(self, blob):
        t = {"uuid": None, "name": None, "desc": None, "parent": None, "proc": None, "thread": None,
             "counter": None, "rank": None}
        for field, _, val in parse_message(blob):
            key = {1: "uuid", 2: "name", 5: "parent", 14: "desc", 3: "proc", 4: "thread", 8: "counter",
                   12: "rank"}.get(field)
            if key in ("name", "desc"):
                t[key] = val.decode("utf-8")
            elif key in ("proc", "thread", "counter"):
                t[key] = {f: v for f, _, v in parse_message(val)}
            elif key:
                t[key] = val
        self.tracks[t["uuid"]] = t


def decode_trace(data):
    """The Chrome-style event dict of a trace made by encode_trace (a metadata instant becomes otherData);
    events come by time, M events first. ts and dur are microseconds again."""
    dec = _Decoder()
    dec.read(data)
    meta, events, other, stacks, counters = [], [], {}, {}, {}
    for uuid, t in sorted(dec.tracks.items()):
        if t["proc"] is not None:
            _add_process_meta(meta, t["proc"])
        elif t["thread"] is not None:
            _add_thread_meta(meta, t["thread"])
    for ts, blob in dec.raw:
        f = {}
        for field, _, v in parse_message(blob):
            f.setdefault(field, []).append(v)
        track = dec.tracks[f[11][0]]
        kind, name = f[9][0], dec.names.get(f.get(10, [0])[0])
        args = dict(_decode_annotation(dec.keys, a) for a in f.get(4, []))
        cat = dec.cats.get(f[3][0]) if 3 in f else None
        us = ts / 1000
        if kind == SLICE_BEGIN:
            stacks.setdefault(track["uuid"], []).append((us, name, cat, args))
        elif kind == SLICE_END:
            start, name, cat, args = stacks[track["uuid"]].pop()
            e = {"ph": "X", "name": name, "pid": track["thread"][1], "tid": track["thread"][2], "ts": start,
                 "dur": us - start, "args": args}
            events.append(dict(e, cat=cat) if cat else e)
        elif kind == INSTANT and track["name"] == METADATA_TRACK_NAME:
            other = args
        elif kind == INSTANT:
            e = _decode_instant(dec, track, name, us, args)
            events.append(dict(e, cat=cat) if cat else e)
        elif kind == COUNTER:
            parent = dec.tracks[track["parent"]]["proc"][1]
            key = (parent, track["desc"], us)
            ev = counters.get(key)
            if ev is None:
                ev = counters[key] = {"ph": "C", "name": track["desc"], "pid": parent, "ts": us, "args": {}}
                events.append(ev)
            ev["args"][track["counter"][6].decode("utf-8")] = struct.unpack("<d", f[44][0])[0]
    return {"traceEvents": meta + events, "otherData": other}


def _decode_instant(dec, track, name, us, args):
    if track["thread"] is not None:
        return {"ph": "i", "s": "t", "name": name, "pid": track["thread"][1], "tid": track["thread"][2],
                "ts": us, "args": args}
    if track["proc"] is None:
        return {"ph": "i", "s": "g", "name": name, "pid": 0, "tid": 0, "ts": us, "args": args}
    return {"ph": "i", "s": "p", "name": name, "pid": track["proc"][1], "tid": 0, "ts": us, "args": args}


def _add_process_meta(meta, proc):
    pid = proc[1]
    if 6 in proc:
        meta.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                     "args": {"name": proc[6].decode("utf-8")}})
    if 3 in proc:
        meta.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                     "args": {"sort_index": proc[3]}})


def _add_thread_meta(meta, thread):
    pid, tid = thread[1], thread[2]
    if 5 in thread:
        meta.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                     "args": {"name": thread[5].decode("utf-8")}})
    if 3 in thread:
        meta.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid,
                     "args": {"sort_index": thread[3]}})


def read_trace(path):
    """The event dict of the trace file at `path`."""
    with open(path, "rb") as fh:
        return decode_trace(fh.read())


def decode_tracks(data):
    """(tracks, events) of a trace whose names are inline, as gem5's coprocessor-trace/2: tracks is
    {uuid: {name, parent, process, counter}} (process and counter are True when the descriptor has one) and
    events [{ts, kind, track, name, args, value}] in file order, ts in ns; value is the counter's integer."""
    tracks, events = {}, []
    for field, _, packet in parse_message(data):
        if field != 1:
            continue
        f = {}
        for k, _, v in parse_message(packet):
            f.setdefault(k, []).append(v)
        for blob in f.get(60, []):
            d = {k: v for k, _, v in parse_message(blob)}
            tracks[d[1]] = {"name": d[2].decode("utf-8") if 2 in d else None, "parent": d.get(5),
                            "process": 3 in d, "counter": 8 in d}
        for blob in f.get(11, []):
            e = {}
            for k, _, v in parse_message(blob):
                e.setdefault(k, []).append(v)
            value = e[30][0] - (1 << 64) if 30 in e and e[30][0] >> 63 else e.get(30, [None])[0]
            events.append({"ts": f.get(8, [0])[0], "kind": e[9][0], "track": e[11][0],
                           "name": e[23][0].decode("utf-8") if 23 in e else None,
                           "args": dict(_decode_annotation({}, a) for a in e.get(4, [])), "value": value})
    return tracks, events
