#!/usr/bin/env python3
"""Convert a TOGSim `--log_level trace` log into a Chrome Trace Event JSON that
opens in Perfetto (https://ui.perfetto.dev) or chrome://tracing as an interactive
timeline (Gantt).

Each instruction becomes one duration slice, grouped per core (pid). Lanes:
  dram-rd -- loads crossing the DRAM bus (read bandwidth)
  dram-wr -- stores crossing the DRAM bus (write bandwidth)
  sa / sa0.. -- COMP compute_type 1 (matmul) / 2 (preload)
  vector  -- COMP compute_type 0 (vector)
Time unit = core cycles. Barriers (MEMORY_BAR/COMPUTE_BAR) are not drawn. A DMA bar
runs from the op's first DRAM response (DRAM_RESP_FIRST, logged by the Core -- so it
captures data moving even while still injecting) to its completion (load: data-ready;
store: finished), serialized per direction so each is one visible bar (packed row =
saturated bus). A compute slice's width is its occupancy (compute_cycle - overlapping).

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

# Perfetto/catapult reserved color names. Slices are tinted per work-item tile, so one
# tile's ops share a color across lanes. 16 names, because a core's tiles stride by
# num_cores and an 8-name palette collapses to 4 colors per core on 2 cores.
_TILE_PALETTE = ["good", "bad", "terrible", "yellow", "olive", "rail_response",
                 "rail_load", "rail_animation", "rail_idle", "thread_state_running",
                 "thread_state_runnable", "thread_state_iowait",
                 "thread_state_uninterruptible", "generic_work", "startup",
                 "vsync_highlight_color"]


def _tile_color(detail):
    m = re.search(r"\btile=(\d+)", detail or "")
    return _TILE_PALETTE[int(m.group(1)) % len(_TILE_PALETTE)] if m else None


_DMA_SHORT = {"MOVIN": "MVIN", "MOVOUT": "MVOUT"}


def _tile_of(detail):
    m = re.search(r"\btile=(-?\d+)", detail or "")
    return m.group(1) if m else "?"


def _label(opcode, detail):
    if opcode == "COMP":
        m = re.search(r"compute_type=(\d+)", detail)
        ct = int(m.group(1)) if m else -1
        return f"T{_tile_of(detail)} {_CT_NAME.get(ct, 'comp')}"
    # DMA: keep each load's OWN identity (addr_name) so the input/weight/K-panel
    # loads stay distinct; tile is conveyed by color (and args), not the name.
    m = re.search(r"addr_name=(\w+)", detail or "")
    who = m.group(1) if m else "?"
    return f"{who} (T{_tile_of(detail)} {_DMA_SHORT.get(opcode, opcode)})"


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
            "issued": None, "finished": None, "resp": None, "dma_issue": None,
            "first_resp": None})
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
        elif tag == "DRAM_RESP_FIRST" and r["first_resp"] is None:  # first data arrived
            r["first_resp"] = cyc
        elif tag == "ASYNC_DMA_ISSUE":   # all requests injected (engine done)
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
      sa     : COMP type 1/2 -- each op on the SA the Core reports (`sa=` field;
               weight-pinned), so lanes auto-split sa0..; rr fallback if absent.
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
        args = {"inst_id": r["iid"], "tile": _tile_of(r["detail"]),
                "issued": r["issued"], "first_data": r["first_resp"],
                "finished": r["finished"], "data_ready": r["resp"]}
        am = re.search(r"addr_name=(\w+)", r["detail"] or "")
        if am:
            args["addr"] = am.group(1)
        ev = {"name": name, "cat": lane, "ph": "X", "ts": ts,
              "dur": max(dur, 1), "pid": core, "tid": lane, "args": args}
        cname = _tile_color(r["detail"])
        if cname:
            ev["cname"] = cname
        events.append(ev)

    def issue_key(r):
        return r["issued"] if r["issued"] is not None else 0

    nsa = max(num_sa, 1)
    for core, u in sorted(by_core.items()):
        # DMA data on the DRAM bus, split by direction: a LOAD's data returns on the
        # response, so its bar is [first response, data-ready]; a STORE's goes out with
        # the request, so its bar is [issued, finished]. Serialized per direction.
        for lane, op, sk, ek in (("dram-rd", "MOVIN", "first_resp", "resp"),
                                 ("dram-wr", "MOVOUT", "issued", "finished")):
            free = 0
            rows = [r for r in u["dma"] if r["opcode"] == op
                    and r[sk] is not None and r[ek] is not None and r[ek] > r[sk]]
            for r in sorted(rows, key=lambda r: r[ek]):
                start = max(r[sk], free)
                free = max(r[ek], start + 1)
                add(core, lane, start, free - start, _label(r["opcode"], r["detail"]), r)
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
        # SA: each op runs on the systolic array the Core reports (the `sa=` field
        # = its weight-pinned / round-robin assignment); fall back to round-robin
        # by issue order for older logs without the field. Each SA is one server.
        rows = sorted(u["sa"], key=issue_key)

        def _sa_of(r, i):
            m = re.search(r"\bsa=(-?\d+)", r["detail"])
            return int(m.group(1)) if (m and int(m.group(1)) >= 0) else (i % nsa)

        max_sa = max([nsa] + [_sa_of(r, i) + 1 for i, r in enumerate(rows)])
        sa_free = [0] * max_sa
        for i, r in enumerate(rows):
            if r["issued"] is None:
                continue
            s = _sa_of(r, i)
            cc, ov = _occ(r["detail"])
            dur = max(cc - ov, 1)
            start = max(r["issued"], sa_free[s])
            sa_free[s] = start + dur
            lane = "sa" if max_sa == 1 else f"sa{s}"
            add(core, lane, start, dur, _label(r["opcode"], r["detail"]), r)

    for c in sorted(cores):
        events.append({"name": "process_name", "ph": "M", "pid": c, "tid": 0,
                       "args": {"name": f"Core {c}"}})
    order = {"dram-rd": 0, "dram-wr": 1,
             "sa": 2, "sa0": 2, "sa1": 3, "sa2": 4, "sa3": 5, "vector": 7}
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
