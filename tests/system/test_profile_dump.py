"""dump_profile turns TOGSim logs into a Perfetto trace on the simulated time axis (kernels, copy, unit and
DRAM counters, instruction slices) and one tile trace per kernel; empty runs stay off the axis."""
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

HERE = os.path.dirname(os.path.abspath(__file__))
TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TORCHSIM_DIR)
sys.path.insert(0, os.path.join(TORCHSIM_DIR, "scripts"))

from pytorchsim.simulator import perfetto_trace  # noqa: E402
import dump_profile  # noqa: E402
import profile_sol  # noqa: E402
import profile_trace  # noqa: E402

DATA = os.path.join(HERE, "data", "profile")
LOG = os.path.join(DATA, "togsim_two_kernels.log")
MHZ = 1751
KERNEL = "pytorchsim_triton_compiler_fused_add_abcd1234"


def _workdir(root, tiles=True):
    """A kernel workdir under `root` with the meta.json of KERNEL and, optionally, a tile timeline."""
    wd = os.path.join(root, "dump", "triton_fixture")
    os.makedirs(os.path.join(wd, "launch-1-1-abcd", "copy_node_1"), exist_ok=True)
    with open(os.path.join(wd, "meta.json"), "w") as fh:
        json.dump({"kernel_name": KERNEL}, fh)
    if tiles:
        shutil.copy(os.path.join(DATA, "tog_tile_timeline.pftrace"), wd)
    return wd


def _log(root, name="run.log", **kw):
    """The fixture log with its container paths pointing into a fresh workdir under `root`."""
    wd = _workdir(root, **kw)
    with open(LOG) as fh:
        text = fh.read().replace("/data/dump/triton_fixture", wd)
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path, wd


def _trace(runs, **kw):
    trace, _ = profile_trace.build_trace(runs, **kw)
    return trace


def _by(trace, **want):
    return [e for e in trace["traceEvents"] if all(e.get(k) == v for k, v in want.items())]


def test_kernel_and_copy_slices_carry_cycles_on_the_simulated_axis():
    """Each launch is an X slice named by its short name; its copy_node runs sit on the copy thread right
    before it and are counted into cycles_incl_copy only."""
    with tempfile.TemporaryDirectory() as root:
        path, wd = _log(root)
        trace = _trace(profile_trace.collect_runs([path]))
    kernel, = _by(trace, ph="X", tid=profile_trace.TID_KERNELS)
    copy, = _by(trace, ph="X", tid=profile_trace.TID_COPY)
    assert kernel["name"] == "fused_add" and copy["name"] == "copy", (kernel["name"], copy["name"])
    got = (copy["args"]["cycles"], kernel["args"]["cycles_excl_copy"], kernel["args"]["cycles_incl_copy"])
    assert got == (400, 649, 1049), got
    assert kernel["args"]["workdir"] == wd and kernel["args"]["inductor_kernel"] == KERNEL
    assert abs(copy["ts"] + copy["dur"] - kernel["ts"]) < 1e-9 and abs(kernel["dur"] - 649 / MHZ) < 1e-9


def test_unit_counters_hold_the_logged_utilization_per_core():
    """A unit counter is named by the unit's short label and holds the % the Core stat window printed."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        trace = _trace(profile_trace.collect_runs([path]))
    names = {e["name"] for e in trace["traceEvents"] if e["ph"] == "C" and e["pid"] == 1}
    want = {"Issue util %", "MISC util %", "MXU util %", "SFU util %", "XLU util %", "Vpu util %"}
    assert names == want, names
    vpu, = _by(trace, ph="C", name="Vpu util %")
    assert vpu["args"] == {"util %": 37.5} and vpu["ts"] == 0


def test_dram_counters_total_by_default_channels_on_request():
    """DRAM has GB/s, utilization % and requests per window; per-channel tracks only with dram_channels."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        runs = profile_trace.collect_runs([path])
        plain, wide = _trace(runs), _trace(runs, dram_channels=True)
    gbps, = _by(plain, ph="C", name="DRAM GB/s")
    assert gbps["args"] == {"GB/s": 16.41} and gbps["pid"] == profile_trace.PID_DRAM
    assert _by(plain, ph="C", name="DRAM requests")[0]["args"] == {"requests": 32}
    assert not _by(plain, ph="C", name="DRAM channel 0 GB/s")
    assert _by(wide, ph="C", name="DRAM channel 0 GB/s")[0]["args"] == {"GB/s": 0.51}


def test_trace_level_log_adds_dma_compute_and_tile_tracks():
    """MOVIN and MOVOUT land on DMA read/write, COMP on a compute lane, TILE_SCHEDULED is a marker."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        trace = _trace(profile_trace.collect_runs([path]))
    read, = _by(trace, ph="X", tid=profile_trace.TID_DMA_READ)
    write, = _by(trace, ph="X", tid=profile_trace.TID_DMA_WRITE)
    assert read["args"]["cycles"] > 0 and write["name"].endswith("MVOUT)"), (read, write)
    lanes = [e for e in trace["traceEvents"] if e.get("cat") == "insn" and e["tid"] >= 10]
    assert [e["args"]["cycles"] for e in lanes] == [12]
    assert _by(trace, ph="i", tid=profile_trace.TID_TILES)[0]["args"] == {"start_cycle": 1, "cycles": 0}


def test_empty_runs_stay_off_the_axis_and_runs_are_marked():
    """A log that scheduled nothing is listed as empty and adds no event; a live run gets a global marker."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        with open(path) as fh:
            idle = re.sub(r"^.*Scheduler.*\n", "", fh.read(), flags=re.M)
        with open(os.path.join(root, "idle.log"), "w") as fh:
            fh.write(idle)
        with open(path) as fh:
            one_cycle = fh.read().replace("Total execution cycles: 1822", "Total execution cycles: 1")
        with open(os.path.join(root, "one_cycle.log"), "w") as fh:
            fh.write(one_cycle)
        trace = _trace(profile_trace.collect_runs([root]))
    other = trace["otherData"]
    assert other["empty_runs"] == ["idle.log", "one_cycle.log"] and other["runs"] == ["run.log"]
    marks = _by(trace, ph="i", s="g")
    assert [m["args"]["log"] for m in marks] == ["run.log"]


def test_runs_are_concatenated_in_execution_order():
    """The second run starts where the first ended: ts offsets add up in microseconds."""
    with tempfile.TemporaryDirectory() as root:
        first, _ = _log(root, "20261113_000000_a.log")
        second, _ = _log(root, "20261113_000001_b.log")
        trace = _trace(profile_trace.collect_runs([root]))
    marks = _by(trace, ph="i", s="g")
    assert [m["args"]["log"] for m in marks] == ["20261113_000000_a.log", "20261113_000001_b.log"]
    assert abs(marks[1]["ts"] - 1822 / MHZ) < 1e-9, marks[1]["ts"]


def test_trace_is_perfetto_clean_and_check_rejects_bad_events():
    """Names come as metadata events; a negative duration, a NaN or a non-numeric counter is refused."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        trace = _trace(profile_trace.collect_runs([path]))
    profile_trace.check_trace(trace)
    meta = {(e["name"], e["pid"]): e["args"] for e in trace["traceEvents"] if e["ph"] == "M"}
    assert meta[("process_name", 1)]["name"] == "NPU core 0"
    assert meta[("process_name", 1000)]["name"] == "DRAM"
    threads = {e["args"]["name"] for e in trace["traceEvents"]
               if e["ph"] == "M" and e["name"] == "thread_name"}
    assert {"Kernels", "copy", "DMA read", "DMA write", "tiles"} <= threads, threads
    for bad in ({"ph": "X", "ts": 0, "dur": -1}, {"ph": "X", "ts": float("nan"), "dur": 1},
                {"ph": "C", "ts": 0, "args": {"v": "x"}}, {"ph": "C", "ts": 0, "args": {}}):
        try:
            profile_trace.check_trace({"traceEvents": [bad]})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_tile_trace_has_a_process_per_row_and_is_skipped_without_a_file():
    """tiles/<triton dir>.pftrace holds the sidecar's rows as processes; a kernel without one writes
    nothing."""
    with tempfile.TemporaryDirectory() as root:
        path, wd = _log(root)
        out = os.path.join(root, "out")
        runs = profile_trace.collect_runs([path])
        dump_profile.write_profile(runs, out)
        tiles = perfetto_trace.read_trace(os.path.join(out, "tiles", "triton_fixture.pftrace"))
        os.remove(os.path.join(wd, "tog_tile_timeline.pftrace"))
        bare = os.path.join(root, "bare")
        dump_profile.write_profile(runs, bare)
        bare_files = sorted(os.listdir(bare))
    names = [e["args"]["name"] for e in tiles["traceEvents"] if e["name"] == "process_name"]
    assert names == ["tile row 0: tile 0 call 0 (16 cycles)", "tile row 2: tile 2 call 1 (14 cycles)"], names
    assert {e["pid"] for e in tiles["traceEvents"] if e["ph"] != "M"} == {1, 3}
    assert bare_files == ["trace.pftrace"], bare_files


def test_tile_timeline_of_another_schema_is_refused():
    with tempfile.TemporaryDirectory() as root:
        _, wd = _log(root)
        perfetto_trace.write_trace(os.path.join(wd, "tog_tile_timeline.pftrace"),
                                   {"traceEvents": [], "otherData": {"schema": "other/1"}})
        try:
            profile_trace.read_tile_timeline(os.path.join(wd, "launch-1-1-abcd", "tog.so"))
        except ValueError as e:
            assert "other/1" in str(e)
            return
    raise AssertionError("accepted a foreign schema")


def test_origin_from_inductor_debug_is_utf8_and_map_path_finds_the_workdir():
    """A model line with non-ASCII text reaches the kernel's args intact; --map-path reads a container
    path from the host prefix."""
    with tempfile.TemporaryDirectory() as root:
        path, wd = _log(root)
        dbg = os.path.join(root, "dbg")
        os.makedirs(dbg)
        with open(os.path.join(dbg, "output_code.py"), "w", encoding="utf-8") as fh:
            fh.write(f"async_compile.triton('k', '''x''', kernel_name='{KERNEL}')\n"
                     f"# Topologically Sorted Source Nodes: [y], Original ATen: [aten.add]\n"
                     f"{KERNEL}.run(a, b)\n")
        with open(os.path.join(dbg, "fx_graph_readable.py"), "w", encoding="utf-8") as fh:
            fh.write("# File: /m/modèle.py:3 in forward\ny = torch.add(x, 1)\n")
        with open(path) as fh:
            mapped = fh.read().replace(wd, "/container/dump/triton_fixture")
        with open(path, "w") as fh:
            fh.write(mapped)
        out = os.path.join(root, "out")
        with redirect_stdout(StringIO()):
            rc = dump_profile.main([path, "--inductor-debug", dbg, "--map-path",
                                    f"/container/dump/triton_fixture={wd}", "-o", out])
        decoded = perfetto_trace.read_trace(os.path.join(out, "trace.pftrace"))
        text = json.dumps(decoded, ensure_ascii=False)
    assert rc == 0
    kernel, = _by(decoded, ph="X", tid=profile_trace.TID_KERNELS)
    assert kernel["args"]["origin"]["lines"] == ["/m/modèle.py:3 in forward"], kernel["args"]
    assert kernel["args"]["origin"]["aten"] == ["aten.add"] and "Ã" not in text


def _window_log(root, vpu2=77.5, gbps2=32.82, copy_end=300):
    """The fixture log with a second Core stat and DRAM window (Vpu `vpu2`, `gbps2` GB/s) and the copy kernel
    ending at `copy_end`."""
    path, wd = _log(root)
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    first = next(i for i, ln in enumerate(lines) if "Core stat" in ln)
    last = next(i for i, ln in enumerate(lines) if "Kernel 0 has completed" in ln)
    block = "".join(lines[first:last]).replace("37.50", f"{vpu2:.2f}").replace("16.41", f"{gbps2:.2f}")
    text = "".join(lines[:last]) + block + "".join(lines[last:])
    with open(path, "w") as fh:
        fh.write(text.replace("finished at cycle 400", f"finished at cycle {copy_end}"))
    return path


def _kernels(path):
    """(copy record, kernel record) of the fixture log's only run."""
    trace, kernel_runs = profile_trace.build_trace(profile_trace.collect_runs([path]))
    copy, kernel = kernel_runs[0]
    return copy, kernel, trace


def test_sol_of_a_kernel_is_the_cycle_weighted_mean_of_the_windows_it_overlaps():
    """Windows 0-200 (Vpu 37.5, DMA 1%, 16.41 GB/s) and 200-400 (Vpu 77.5, 32.82 GB/s): a copy ending at 300
    overlaps them 200:100 cycles (Vpu 50.83); the kernel's incl-copy span 0-1049 overlaps 200:200 (57.5)."""
    with tempfile.TemporaryDirectory() as root:
        copy, kernel, _ = _kernels(_window_log(root))
    assert copy["span"] == (0, 300) and kernel["span"] == (0, 1049)
    assert abs(copy["sol"]["util"]["Vpu"] - (37.5 * 200 + 77.5 * 100) / 300) < 1e-9
    assert abs(kernel["sol"]["compute_pct"] - 57.5) < 1e-9 and kernel["sol"]["compute_unit"] == "Vpu"
    assert kernel["sol"]["util"]["MXU"] == 0.0 and kernel["sol"]["source"] == "windows"
    assert abs(kernel["sol"]["dma_pct"] - 1.0) < 1e-9
    window = 200 * 1751 / 3205
    want = (16.41 * window + 32.82 * window) / (2 * window)
    assert abs(kernel["sol"]["dram_gbps"] - want) < 1e-9
    assert abs(kernel["sol"]["dram_pct"] - want / 1638.4 * 100) < 1e-9
    assert kernel["sol"]["verdict"] == "latency/dependency-bound"


def test_sol_falls_back_to_the_whole_run_table_when_no_window_overlaps():
    """Without any window the Core table printed at the end and the per-channel DRAM averages answer: Vpu
    0.11 %, DMA 4/1822 cycles, 0.22 GB/s."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        with open(path) as fh:
            lines = fh.read().splitlines(keepends=True)
        first = next(i for i, ln in enumerate(lines) if "Core stat" in ln)
        last = next(i for i, ln in enumerate(lines) if "Kernel 0 has completed" in ln)
        with open(path, "w") as fh:
            fh.write("".join(lines[:first] + lines[last:]))
        _, kernel, _ = _kernels(path)
    sol = kernel["sol"]
    assert sol["source"] == "run-total" and sol["compute_unit"] == "Vpu"
    assert abs(sol["compute_pct"] - 0.11) < 1e-9
    assert abs(sol["dma_pct"] - 4 / 1822 * 100) < 1e-9 and abs(sol["dram_gbps"] - 0.22) < 1e-9


def test_verdict_follows_the_thresholds_and_the_logged_numbers():
    """BOUND and LOW decide: a unit at 85 % is compute-bound (named), 1200 GB/s of 1638.4 is memory-bound, a
    DMA 75 % busy under 1 % DRAM is DMA-latency-bound; the fixture's low numbers are latency/dependency."""
    assert (profile_sol.BOUND, profile_sol.LOW) == (60.0, 20.0)
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        with open(path) as fh:
            text = fh.read()
        got = {}
        for name, old, new in (("compute", "Vpu utilization(%): 37.50", "Vpu utilization(%): 85.00"),
                               ("memory", "16.41 GB/s aggregate", "1200.00 GB/s aggregate"),
                               ("dma", "DMA active_cycles: 2, DMA idle_cycles: 198",
                                "DMA active_cycles: 150, DMA idle_cycles: 50")):
            with open(path, "w") as fh:
                fh.write(text.replace(old, new))
            got[name] = _kernels(path)[1]["sol"]["verdict"]
    assert got == {"compute": "compute-bound (Vpu)", "memory": "memory-bound (DRAM)",
                   "dma": "DMA-latency-bound"}, got
    assert profile_sol.get_verdict(70, "MXU", 65, 10) == "compute-bound (MXU)"
    assert profile_sol.get_verdict(50, "MXU", 65, 10) == "memory-bound (DRAM)"
    assert profile_sol.get_verdict(10, "MXU", 30, 90) == "latency/dependency-bound"
    assert profile_sol.get_verdict(None, None, None, None) == "unknown"


def test_kernel_slices_carry_sol_args_and_where_to_go_next():
    """The kernel slice has the speed-of-light numbers and relative tile_trace, source and dump_dir paths;
    start_cycle and cycles sit on kernel, copy, DMA, compute and tile events."""
    with tempfile.TemporaryDirectory() as root:
        path, wd = _log(root)
        with open(os.path.join(wd, f"{KERNEL}_triton.py"), "w") as fh:
            fh.write("@triton.jit\n")
        out = os.path.join(root, "profile")
        trace, _, _, _ = dump_profile.write_profile(profile_trace.collect_runs([path]), out)
        tiles = perfetto_trace.read_trace(os.path.join(out, "tiles", "triton_fixture.pftrace"))
    kernel, = _by(trace, ph="X", tid=profile_trace.TID_KERNELS)
    args = kernel["args"]
    assert args["sol_compute_pct"] == 37.5 and args["sol_compute_unit"] == "Vpu" and args["util_Vpu"] == 37.5
    assert args["sol_dram_pct"] == 1.0 and args["dma_busy_pct"] == 1.0 and args["util_MXU"] == 0.0
    assert args["verdict"] == "latency/dependency-bound" and args["sol_span_cycles"] == 1049
    assert args["tile_trace"] == "tiles/triton_fixture.pftrace"
    assert args["source"] == f"../dump/triton_fixture/{KERNEL}_triton.py"
    assert args["dump_dir"] == "../dump/triton_fixture"
    slices = [e for e in trace["traceEvents"] if e["ph"] in ("X", "i") and e["pid"] != 0]
    assert slices and all({"start_cycle", "cycles"} <= set(e["args"]) for e in slices), slices
    copy, = _by(trace, ph="X", tid=profile_trace.TID_COPY)
    assert (kernel["args"]["start_cycle"], copy["args"]["start_cycle"]) == (400, 0)
    rows = [e for e in tiles["traceEvents"] if e["ph"] == "X"]
    assert sorted((round(e["ts"] * 1000), round(e["dur"] * 1000)) for e in rows) == [(13, 1), (15, 1)]


def test_trace_metadata_names_the_machine_and_the_time_axis():
    """otherData says which machine, the core MHz and what one microsecond is, and states the verdict rule."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        trace = _trace(profile_trace.collect_runs([path]))
    other = trace["otherData"]
    assert other["machine"] == "tpuv6e.yml" and other["core_mhz"] == MHZ
    assert other["time_axis"] == "simulated runs concatenated in log order; 1 us = 1751 cycles"
    assert "utilization-based" in other["verdict_note"].lower() and "60%" in other["verdict_note"]


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


def test_trace_that_cannot_nest_is_refused_before_encoding():
    """Two slices on one thread that overlap without containing each other have no binary form; the check
    names them, and a thread of sequential or nested slices passes."""
    def slice_(ts, dur):
        return {"ph": "X", "name": f"s{ts}", "pid": 1, "tid": 1, "ts": ts, "dur": dur, "args": {}}
    profile_trace.check_trace({"traceEvents": [slice_(0, 10), slice_(2, 3), slice_(10, 5)]})
    try:
        profile_trace.check_trace({"traceEvents": [slice_(0, 10), slice_(5, 10)]})
    except ValueError as exc:
        assert "overlaps" in str(exc)
        return
    raise AssertionError("accepted overlapping slices")


def test_written_trace_is_the_encoded_event_model_and_names_the_axis():
    """trace.pftrace decodes to the very events build_trace made (ts to the ns) with the axis sentence in
    otherData, and no trace.json is written."""
    with tempfile.TemporaryDirectory() as root:
        path, _ = _log(root)
        out = os.path.join(root, "out")
        trace, _, _, _ = dump_profile.write_profile(profile_trace.collect_runs([path]), out)
        back = perfetto_trace.read_trace(os.path.join(out, "trace.pftrace"))
        files = sorted(os.listdir(out))
    assert files == ["tiles", "trace.pftrace"], files
    assert back["otherData"] == trace["otherData"]
    assert back["otherData"]["time_axis"] == "simulated runs concatenated in log order; 1 us = 1751 cycles"
    assert [json.loads(e) for e in _canon(back)] == [json.loads(e) for e in _canon(trace)]


def test_tile_trace_names_its_kernel_and_keeps_the_timeline():
    """build_tile_trace adds the kernel to the tile timeline's metadata and nothing else; the result is
    non-overlapping per thread, so it encodes."""
    timeline = profile_trace.read_tile_timeline(None, os.path.join(DATA, "tog_tile_timeline.pftrace"))
    trace = profile_trace.build_tile_trace(timeline, "k")
    profile_trace.check_trace(trace)
    assert trace["otherData"] == {**timeline["otherData"], "kernel": "k"}
    assert trace["traceEvents"] == timeline["traceEvents"]


def test_cli_without_logs_fails():
    with tempfile.TemporaryDirectory() as root, redirect_stderr(StringIO()):
        assert dump_profile.main([os.path.join(root, "none"), "-o", os.path.join(root, "o")]) == 1
