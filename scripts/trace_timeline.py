#!/usr/bin/env python3
"""Convert a TOGSim `--log_level trace` log into a Chrome Trace Event JSON that
opens in Perfetto (https://ui.perfetto.dev) or chrome://tracing as an interactive
timeline (Gantt).

Each instruction becomes one duration slice on one of 3 per-core lanes:
  dma     -- MOVIN / MOVOUT
  sa      -- COMP compute_type 1 (matmul) / 2 (preload)
  vector  -- COMP compute_type 0 (vector)
grouped per core (pid). Time unit = core cycles. Barriers (MEMORY_BAR/COMPUTE_BAR)
are not drawn. A compute slice's width is its compute_cycle (the op's own latency),
not issue->finish (which balloons under pipeline backlog); a DMA slice is the
actual transfer ASYNC_DMA_ISSUE -> data-ready.

Usage:
  bin/Simulator --config <yml> --trace_so <so> --cycle_table <tsv> --log_level trace \
      2>&1 | python scripts/trace_timeline.py -o timeline.json
  # or
  python scripts/trace_timeline.py trace.log -o timeline.json
Then drag timeline.json into https://ui.perfetto.dev .
"""
import argparse
import json
import re
import sys

# [cycle][Core C][TAG ][INST_ID=N] OPCODE (detail...)
_LINE = re.compile(
    r"\[(\d+)\]\[Core (\d+)\]\[([A-Z_]+)\s*\](?:\[INST_ID=(-?\d+)\])?\s*(\w+)?(.*)")

# Only 3 lanes per core. Barriers are dropped (see _HIDE).
_LANE = {"MOVIN": "dma", "MOVOUT": "dma"}
_HIDE = {"MEMORY_BAR", "COMPUTE_BAR", "TILE_BEGIN", "TILE_END"}
_CT_NAME = {0: "vector", 1: "matmul", 2: "preload"}


def _label(opcode, detail):
    if opcode == "COMP":
        m = re.search(r"compute_type=(\d+)", detail)
        ct = int(m.group(1)) if m else -1
        return _CT_NAME.get(ct, "comp")
    m = re.search(r"addr_name=(\w+)", detail)
    return f"{opcode} {m.group(1)}" if m else opcode


def _lane(opcode, detail):
    if opcode == "COMP":
        m = re.search(r"compute_type=(\d+)", detail)
        ct = int(m.group(1)) if m else -1
        return "vector" if ct == 0 else "sa"
    return _LANE.get(opcode, "dma")


def parse(lines):
    # key = (core, inst_id) -> record
    insts = {}
    for ln in lines:
        m = _LINE.search(ln)
        if not m:
            continue
        cyc, core, tag, iid, opcode, detail = m.groups()
        if iid is None or opcode is None:
            continue
        cyc, core, iid = int(cyc), int(core), int(iid)
        key = (core, iid)
        r = insts.setdefault(key, {
            "core": core, "iid": iid, "opcode": opcode, "detail": detail,
            "issued": None, "finished": None, "resp": None, "dma_issue": None})
        if not r["opcode"] or r["opcode"] == opcode:
            r["opcode"] = opcode
            if detail.strip():
                r["detail"] = detail
        if tag == "INST_ISSUED" and r["issued"] is None:
            r["issued"] = cyc
        elif tag == "INST_FINISHED":
            r["finished"] = cyc
        elif tag == "DRAM_RESP_DONE":
            r["resp"] = cyc
        elif tag == "ASYNC_DMA_ISSUE":   # actual transfer start (DMA engine busy)
            r["dma_issue"] = cyc
    return insts


def _occ(detail):
    """(compute_cycle, overlapping_cycle) from a COMP detail string."""
    cc = re.search(r"compute_cycle=(\d+)", detail)
    ov = re.search(r"overlapping_cycle=(\d+)", detail)
    return (int(cc.group(1)) if cc else 0, int(ov.group(1)) if ov else 0)


def to_chrome(insts, num_sa=1):
    """Model each hardware unit as a server and replay its ops in issue order, so
    real idle gaps (bubbles) show and slices don't nest:
      dma    : MOVIN/MOVOUT -- 1 DMA engine; slice = actual transfer
               (ASYNC_DMA_ISSUE -> data-ready).
      vector : COMP type 0  -- 1 VPU.
      sa     : COMP type 1/2 -- num_sa systolic arrays, round-robin by issue order.
    A compute slice's width is compute_cycle - overlapping_cycle (its occupancy =
    latency minus the tail that overlaps the next op), starting when the unit
    actually picks it up: start = max(issue, unit_free). num_sa>1 -> lanes sa0.. ."""
    by_core = {}
    for r in insts.values():
        op, detail, core = r["opcode"], r["detail"], r["core"]
        if op in _HIDE:
            continue
        u = by_core.setdefault(core, {"dma": [], "vector": [], "sa": []})
        if op == "COMP":
            m = re.search(r"compute_type=(\d+)", detail)
            ct = int(m.group(1)) if m else -1
            u["vector" if ct == 0 else "sa"].append(r)
        else:
            u["dma"].append(r)

    events, lanes, cores = [], set(), set()

    def add(core, lane, ts, dur, name, r):
        lanes.add((core, lane))
        cores.add(core)
        events.append({"name": name, "cat": lane, "ph": "X", "ts": ts,
                       "dur": max(dur, 1), "pid": core, "tid": lane,
                       "args": {"inst_id": r["iid"], "issued": r["issued"],
                                "finished": r["finished"], "data_ready": r["resp"]}})

    def issue_key(r):
        return r["issued"] if r["issued"] is not None else 0

    nsa = max(num_sa, 1)
    for core, u in sorted(by_core.items()):
        # DMA engine: one server, serialized. A load occupies it only while INJECTING
        # requests -- [INST_ISSUED, ASYNC_DMA_ISSUE] -- not the response tail. So a load
        # blocked on a full spad leaves a real idle gap = the SRAM throttle stalling it.
        free = 0
        for r in sorted(u["dma"], key=issue_key):
            start = r["issued"] if r["issued"] is not None else r["dma_issue"]
            end = r["dma_issue"]
            if end is None:  # sync dma / store: no async-issue marker
                end = r["resp"] if r["resp"] is not None else r["finished"]
            if start is None:
                continue
            if end is None or end < start:
                end = start + 1
            start = max(start, free)
            free = max(end, start + 1)
            add(core, "dma", start, free - start, _label(r["opcode"], r["detail"]), r)
        # VPU: one server; slice = occupancy (compute_cycle - overlapping_cycle).
        free = 0
        for r in sorted(u["vector"], key=issue_key):
            if r["issued"] is None:
                continue
            cc, ov = _occ(r["detail"])
            dur = max(cc - ov, 1)
            start = max(r["issued"], free)
            free = start + dur
            add(core, "vector", start, dur, "vector", r)
        # SA: num_sa servers, round-robin in issue order (mirrors the Core's rr).
        sa_free = [0] * nsa
        for i, r in enumerate(sorted(u["sa"], key=issue_key)):
            if r["issued"] is None:
                continue
            s = i % nsa
            cc, ov = _occ(r["detail"])
            dur = max(cc - ov, 1)
            start = max(r["issued"], sa_free[s])
            sa_free[s] = start + dur
            lane = "sa" if nsa == 1 else f"sa{s}"
            add(core, lane, start, dur, _label(r["opcode"], r["detail"]), r)

    for c in sorted(cores):
        events.append({"name": "process_name", "ph": "M", "pid": c, "tid": 0,
                       "args": {"name": f"Core {c}"}})
    order = {"dma": 0, "sa": 1, "sa0": 1, "sa1": 2, "sa2": 3, "sa3": 4, "vector": 8}
    for c, lane in sorted(lanes, key=lambda x: (x[0], order.get(x[1], 5))):
        events.append({"name": "thread_name", "ph": "M", "pid": c, "tid": lane,
                       "args": {"name": lane}})
        events.append({"name": "thread_sort_index", "ph": "M", "pid": c, "tid": lane,
                       "args": {"sort_index": order.get(lane, 5)}})
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="trace log file (default: stdin)")
    ap.add_argument("-o", "--out", default="timeline.json")
    ap.add_argument("-s", "--num-sa", type=int, default=1,
                    help="systolic arrays per core (num_systolic_array_per_core); "
                         ">1 splits into sa0..saN-1 lanes")
    a = ap.parse_args(argv[1:])
    src = open(a.input) if a.input else sys.stdin
    insts = parse(src)
    trace = to_chrome(insts, a.num_sa)
    with open(a.out, "w") as fh:
        json.dump(trace, fh)
    n = sum(1 for e in trace["traceEvents"] if e["ph"] == "X")
    sys.stderr.write(f"wrote {a.out}: {n} slices -> open in https://ui.perfetto.dev\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
