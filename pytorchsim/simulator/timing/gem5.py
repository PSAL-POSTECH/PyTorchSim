"""gem5 over the compiler's cycle binary, one sampled tile per dump, and the tables TOGSim reads
beside the TOG from those dumps: tog_tile_cycles.tsv (cycle, overlapping) and tog_unit_table.tsv; with
TORCHSIM_GEM5_TRACE=1 also tog_tile_timeline.pftrace, the coprocessor trace split by cycle-table row."""

import os
import re
import subprocess

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge
from pytorchsim.simulator import perfetto_trace, wallclock

logger = config.setup_logger()

TRACE_FILE = "coprocessor_trace.pftrace"
TRACE_SCHEMA = "coprocessor-trace/2"
TS_UNIT = "cycle (1 ns per cycle)"


def run_cycle_binary(workdir, meta):
    """The m5out of the compiler's cycle ELF run under gem5, or None when gem5 could not run it."""

    kernel_name = meta["kernel_name"]
    elf = contract.artifact(workdir, "cycle_elf")
    if elf is None:
        logger.warning("[Gem5] %s has no cycle_elf; it was compiled without --tog", workdir)
        return None
    try:
        with wallclock.span(wallclock.GEM5_RUN, kernel_name):
            return _simulate(
                elf, workdir, contract.kernel_object(workdir), silent_mode=True)
    except Exception as e:
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


def _simulate(target_binary, obj_dir, manifest, silent_mode=False):
    """gem5 over one tile of the kernel object in `obj_dir`, under what it was compiled for."""
    dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
    machine = compiler_bridge.compiled_for(manifest)
    run = compiler_bridge.run_options(obj_dir, manifest)
    gem5_cmd = [compiler_bridge.toolchain().GEM5, "-r", "--stdout-file=sto.log", "-d", dir_path,
                run["gem5_script"], "-c", target_binary,
                "--model", run["model"],
                "--machine-config", run["machine_config"],
                "--vlane", str(machine.vector_lanes), "--vlen", str(machine.vlen_bits)]
    if config.CONFIG_GEM5_TRACE:
        gem5_cmd.append(f"--coprocessor-trace-file={TRACE_FILE}")

    if not silent_mode:
        logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
        logger.info("[Gem5] Gem5 simulation started")

    proc = subprocess.run(gem5_cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(_gem5_failure(proc, os.path.join(dir_path, "sto.log")))

    return dir_path


#: What gem5 says when it dies, as opposed to the libc backtrace that follows it.
_GEM5_SIGNAL = re.compile(
    r"^(?:.*\b(?:panic|fatal):.*|.*Assertion.*|Program aborted at tick.*)$", re.M)


def _gem5_failure(proc, log_path):
    """The message a failed gem5 run raises, from the log it redirects to: -r sends stdout and
    stderr to --stdout-file, so the pipes are empty and the return code alone says nothing."""
    how = (f"signal {-proc.returncode}" if proc.returncode < 0
           else f"exit {proc.returncode}")
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = (proc.stderr or "") + (proc.stdout or "")
    hits = [h.strip() for h in _GEM5_SIGNAL.findall(text)]
    if not hits:
        hits = [l.strip() for l in text.strip().splitlines()[-3:]]
    if not hits:
        hits = [f"gem5 left no diagnostic; see {log_path}"]
    return (f"Gem5 Simulation Failed ({how}); full log in {log_path}\n  "
            + "\n  ".join(h[:300] for h in hits[:3]))


def tile_tables(m5out, overlap_offsets):
    """The cycle table and the unit rows of a gem5 run: a row per dump the compiler's gem5_samples
    reads, in the order it returns them; whatever that reader refuses is raised, not sampled around."""
    dumps = compiler_bridge.compiler_contract("gem5_samples").read(m5out, len(overlap_offsets))
    return cycle_table(overlap_offsets, dumps), unit_rows(dumps)


def write_tile_timeline(m5out, table, labels, path):
    """tog_tile_timeline.pftrace from the coprocessor trace gem5 left in `m5out`: one process per cycle-table
    row (`labels` are its (tile, call)), ts in cycles from the tile's reset; None when tracing was off."""
    trace = os.path.join(m5out, TRACE_FILE)
    if not os.path.isfile(trace):
        return None
    with open(trace, "rb") as fh:
        data = fh.read()
    rows = split_trace_by_tile(data, [c for c, _ in table])
    perfetto_trace.write_trace(path, build_timeline_trace(rows, labels))
    return path


def split_trace_by_tile(data, cycles):
    """[{row, cycles, insns, counters}] from a coprocessor-trace/2 file: each dump closes the window opened
    by the latest reset; slices are clipped to it, counters carry their held value in and all are rebased to
    the window start. Another schema, a window count or a window length that is not `cycles` raises."""
    tracks, events = perfetto_trace.decode_tracks(data)
    stats = [e for e in events if tracks[e["track"]]["name"] == "stats"]
    schema = next((e["args"].get("schema") for e in stats if e["name"] == "schema"), None)
    if schema != TRACE_SCHEMA:
        raise ValueError(f"coprocessor trace schema {schema!r}, expected {TRACE_SCHEMA}")
    windows, start = [], None
    for e in sorted((e for e in stats if e["name"] in ("reset", "dump")), key=lambda e: e["ts"]):
        if e["name"] == "reset":
            start = e["ts"]
        elif start is not None:
            windows.append((start, e["ts"]))
            start = None
    if len(windows) != len(cycles):
        raise ValueError(f"coprocessor trace has {len(windows)} tile windows, "
                         f"the cycle table {len(cycles)} rows")
    slices, counters = _pair_slices(tracks, events), _get_counter_events(tracks, events)
    rows = []
    for row, ((lo, hi), want) in enumerate(zip(windows, cycles)):
        if hi - lo != int(want):
            raise ValueError(f"coprocessor trace window {row} is {hi - lo} cycles, "
                             f"the cycle table says {want}")
        rows.append({"row": row, "cycles": hi - lo, "insns": _clip_slices(slices, lo, hi),
                     "counters": _clip_counters(counters, lo, hi)})
    return rows


def _pair_slices(tracks, events):
    """[(lane, name, begin, end, args)] of the begin/end pairs on the instruction lanes, in file order."""
    open_, out = {}, []
    for e in events:
        t = tracks[e["track"]]
        if t["counter"] or t["name"] == "stats" or t["process"]:
            continue
        if e["kind"] == perfetto_trace.SLICE_BEGIN:
            open_[e["track"]] = e
        elif e["kind"] == perfetto_trace.SLICE_END:
            b = open_.pop(e["track"])
            out.append((t["name"], b["name"], b["ts"], e["ts"], b["args"]))
    return out


def _get_counter_events(tracks, events):
    """{counter name: [(ts, value)]} sorted by ts, from the counter tracks."""
    out = {}
    for e in events:
        if e["kind"] == perfetto_trace.COUNTER:
            out.setdefault(tracks[e["track"]]["name"], []).append((e["ts"], e["value"]))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in out.items()}


def _clip_slices(slices, lo, hi):
    """The slices overlapping [lo, hi) clipped to it, as (lane, name, ts - lo, dur, args), by start."""
    out = []
    for lane, name, a, b, args in slices:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append((lane, name, a - lo, b - a, args))
    return sorted(out, key=lambda s: s[2])


def _clip_counters(counters, lo, hi):
    """[(name, ts - lo, value)]: each counter's value at `lo` carried in at 0, then its changes inside."""
    out = []
    for name, points in counters.items():
        held = [v for ts, v in points if ts <= lo]
        if held:
            out.append((name, 0, held[-1]))
        out += [(name, ts - lo, v) for ts, v in points if lo < ts < hi]
    return sorted(out, key=lambda c: c[1])


def build_timeline_trace(rows, labels):
    """The event dict of a tile timeline: a process per row named by its row, tile, call and cycles, a thread
    per instruction lane, a counter track per counter; ts in us, 1 cycle = 1 ns."""
    events = []
    for r in rows:
        pid = r["row"] + 1
        tile, call = labels[r["row"]]
        events += [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {
                        "name": f"tile row {r['row']}: tile {tile} call {call} ({r['cycles']} cycles)"}},
                   {"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                    "args": {"sort_index": r["row"]}}]
        lanes = {}
        for lane, name, ts, dur, args in r["insns"]:
            if lane not in lanes:
                lanes[lane] = len(lanes) + 1
                events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": lanes[lane],
                               "args": {"name": lane}})
            events.append({"ph": "X", "cat": "insn", "name": name, "pid": pid, "tid": lanes[lane],
                           "ts": ts / 1000, "dur": dur / 1000, "args": args})
        events += [{"ph": "C", "cat": "unit", "name": name, "pid": pid, "ts": ts / 1000,
                    "args": {"delta": value}} for name, ts, value in r["counters"]]
    return {"traceEvents": events, "otherData": {
        "schema": perfetto_trace.TILE_TIMELINE_SCHEMA, "ts_unit": TS_UNIT, "rows": len(rows)}}


def read_tile_labels(m5out, n_tiles):
    """[(tile, call)] per cycle-table row, from the dumps the compiler's gem5_samples reads."""
    dumps = compiler_bridge.compiler_contract("gem5_samples").read(m5out, n_tiles)
    return [(d["tile"], d["call"]) for d in dumps]


def cycle_table(overlap_offsets, dumps):
    """[(cycle, overlapping_cycle), ...], one per gem5 dump in row order. The compiler states each
    tile's overlap offset (None where nothing of it overlaps), and each dump names its tile."""
    return [(int(d["cycles"]), 0 if (off := overlap_offsets[d["tile"]]) is None
             else max(int(d["cycles"]) - int(off), 0))
            for d in dumps]


def unit_rows(dumps):
    """TOGSim's --unit_table rows, one per table row and unit, from gem5's dumps in row order:
    (row, unit, unit_of_work, capacity per cycle, admitted)."""
    rows = []
    for row, dump in enumerate(dumps):
        for name, u in dump["units"].items():
            if "admitted" not in u or "capacity" not in u:
                raise ValueError(f"gem5 dump of tile {dump['tile']} call {dump['call']}: unit {name} "
                                 f"lacks admitted or capacity")
            rows.append((row, name, u["unit_of_work"], int(u["capacity"]), int(u["admitted"])))
    return rows


def dump_cycle_table_tsv(table, path, origins=None):
    """cycle<TAB>overlapping per line, in row order, for TOGSim's loader.

    origins is appended as a trailing comment line, which the loader stops at.
    """
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path


