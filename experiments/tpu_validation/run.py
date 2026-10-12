"""Simulate the shapes of a TPU table through PyTorchSim, then compare with the TPU; resumable, parallel.

    python run.py gemm    --tag g1 [--keep KxN=768x768,2048x3072]
    python run.py softmax --tag s1 [--keep C=128,512,2048,8192] [--drop R=1]
    python run.py softmax --tag s2 --set vpu_num_lanes=128 --set core_freq_mhz=1750

Shapes come from the tpu-perf-archive table data/v6e/<op>/torchxla.csv (cloned on first use, or
$TPU_PERF_ARCHIVE), smallest work first. Everything of a
tag lives in <out>/<tag>/:

    config.yml    the config every shape ran on: --config with $TORCHSIM_DIR expanded and --set applied
    meta.json     op, filters, frontend / compiler / archive commits, every invocation
    results.csv   one row per finished shape, written as it finishes, so a rerun skips what is done
    failed.csv    shapes that failed or timed out; they are retried on the next run
    jobs/<shape>/ stdout.log.gz of the shape (stdout.log if it failed)
    compare.csv   results x TPU table (compare.py, run at the end)

sim_ns = cycles * 1000 / core_freq_mhz of config.yml.
"""
import argparse
import csv
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import compare
from ops import OPS, table_path

HERE = os.path.dirname(os.path.abspath(__file__))
OP = KEYS = TAG = CONFIG = CORE_MHZ = None          # set in __main__
STATS = ["cycles", "sim_ns", "dram_reads", "dram_writes", "sa_util_pct", "wall_s"]
LOCK = threading.Lock()
PROCS = set()

CYCLES = re.compile(r"Total execution cycles: (\d+)")
DRAM = re.compile(r"channels 0\.\.\d+ combined .*\| (\d+) reads, (\d+) writes")
SA = re.compile(r"Systolic(?: array \[\d+\])? utilization\(%\): ([\d.]+)")


def build_config(src, sets):
    text = os.path.expandvars(open(src).read())
    for s in sets:
        k, v = s.split("=", 1)
        text, n = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", text, flags=re.M)
        if not n:
            text += f"{k}: {v}\n"
    return text


def parse_filters(specs):
    """'KxN=768x768,2048x3072' -> ((1, 2), {(768, 768), (2048, 3072)})"""
    out = []
    for spec in specs:
        cols, vals = spec.split("=", 1)
        out.append((tuple(KEYS.index(c) for c in cols.split("x")),
                    {tuple(map(int, v.split("x"))) for v in vals.split(",")}))
    return out


def hits(shape, f):
    return tuple(shape[i] for i in f[0]) in f[1]


def sha(path):
    try:
        git = ["git", "-C", path]
        head = subprocess.check_output(git + ["rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(git + ["status", "--porcelain", "-uno"], text=True, stderr=subprocess.DEVNULL).strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def done():
    p = os.path.join(TAG, "results.csv")
    if not os.path.exists(p):
        return set()
    return {tuple(int(r[k]) for k in KEYS) for r in csv.DictReader(open(p))}


def append(name, fields, row):
    with LOCK:
        path = os.path.join(TAG, name)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fields, lineterminator="\n")
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def run(shape, timeout):
    name = "x".join(map(str, shape))
    job = os.path.join(TAG, "jobs", name)
    shutil.rmtree(job, ignore_errors=True)
    os.makedirs(job)
    env = dict(os.environ, TOGSIM_CONFIG=CONFIG, TORCHSIM_TIMING_MODE="1",
               TORCHSIM_DUMP_PATH=os.path.join(job, "dump"), TORCHSIM_LOG_PATH=os.path.join(job, "log"),
               TORCHINDUCTOR_CACHE_DIR=os.path.join(job, "inductor"))
    t0 = time.time()
    log = os.path.join(job, "stdout.log")
    with open(log, "w") as f:
        # its own process group: a timeout has to take the simulators the script spawned with it
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "sim_op.py"), OP, *map(str, shape)],
                             cwd=job, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        with LOCK:
            PROCS.add(p)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
        with LOCK:
            PROCS.discard(p)
    wall = round(time.time() - t0)
    text = open(log, errors="replace").read()
    for sub in ("dump", "inductor"):                # big and rebuildable; the log is the record
        shutil.rmtree(os.path.join(job, sub), ignore_errors=True)
    c, d = CYCLES.search(text), DRAM.search(text)
    if rc == 0 and c and "Simulation Done" in text:
        cyc = int(c.group(1))
        tail = text[text.rfind("===== Instructions count ====="):]      # the final block, not the periodic ones
        sa = [float(x) for x in SA.findall(tail)]
        append("results.csv", list(KEYS) + STATS,
               dict(zip(KEYS, shape), cycles=cyc, sim_ns=round(cyc * 1000 / CORE_MHZ),
                    dram_reads=d.group(1) if d else "", dram_writes=d.group(2) if d else "",
                    sa_util_pct=round(sum(sa) / len(sa), 2) if sa else "", wall_s=wall))
        with open(log, "rb") as fi, gzip.open(log + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(log)
        print(f"ok   {name}: {cyc} cycles ({wall}s)", flush=True)
    else:
        last = [l for l in text.splitlines() if l.strip()][-1:] or [""]
        append("failed.csv", list(KEYS) + ["rc", "wall_s", "last"],
               dict(zip(KEYS, shape), rc=rc, wall_s=wall, last=last[0][:200]))
        print(f"FAIL {name}: rc={rc} ({wall}s) {last[0][:120]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", choices=sorted(OPS))
    ap.add_argument("--tag", required=True, help="names the run: <out>/<tag>/")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "tpuv6e_nocache.yml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config line, e.g. --set vpu_num_lanes=128 (repeatable)")
    ap.add_argument("--keep", action="append", default=[], metavar="COLS=VALUES",
                    help="only these shapes, e.g. C=128,512 or KxN=768x768,2048x3072 (repeatable: all must match)")
    ap.add_argument("--drop", action="append", default=[], metavar="COLS=VALUES", help="skip these shapes, e.g. R=1")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=2 * 3600, help="seconds per shape")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"), help="where the tag directories live")
    ap.add_argument("--table", help="TPU table, default data/v6e/<op>/torchxla.csv of the archive")
    a = ap.parse_args()
    if "TORCHSIM_DIR" not in os.environ:
        raise SystemExit("TORCHSIM_DIR is not set: source your PyTorchSim environment first")
    OP, KEYS, TAG = a.op, OPS[a.op]["keys"], os.path.join(os.path.abspath(a.out), a.tag)
    table = os.path.abspath(a.table or table_path(OP))
    text = build_config(a.config, a.set)
    CONFIG, meta_path = os.path.join(TAG, "config.yml"), os.path.join(TAG, "meta.json")
    here = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=" ".join(sys.argv[1:]))
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta["op"] != OP or open(CONFIG).read() != text:
            raise SystemExit(f"{TAG} already holds a {meta['op']} run on a different config; pick another --tag")
        meta["invocations"].append(here)
    else:
        os.makedirs(TAG, exist_ok=True)
        open(CONFIG, "w").write(text)
        compile_dir = os.environ.get("TORCHSIM_COMPILE_DIR", "")
        meta = dict(op=OP, table=table, table_sha=sha(os.path.dirname(table)), config_from=os.path.abspath(a.config),
                    set=a.set, torchsim_dir=os.environ["TORCHSIM_DIR"], torchsim_sha=sha(os.environ["TORCHSIM_DIR"]),
                    compile_dir=compile_dir, compile_sha=sha(compile_dir) if compile_dir else "unknown",
                    invocations=[here])
    json.dump(meta, open(meta_path, "w"), indent=1)
    CORE_MHZ = int(re.search(r"^core_freq_mhz:\s*(\d+)", text, re.M).group(1))

    keep, drop = parse_filters(a.keep), parse_filters(a.drop)
    shapes = sorted({tuple(int(r[k]) for k in KEYS) for r in csv.DictReader(open(table))},
                    key=lambda s: (OPS[OP]["size"](s), s))
    finished = done()
    todo = [s for s in shapes if s not in finished and all(hits(s, f) for f in keep) and not any(hits(s, f) for f in drop)]
    print(f"{OP}: {len(todo)} shapes to run, {a.jobs} at a time, core {CORE_MHZ} MHz -> {TAG}", flush=True)
    try:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda s: run(s, a.timeout), todo))
    finally:
        for p in list(PROCS):                       # Ctrl-C: the shapes are in their own groups
            os.killpg(p.pid, signal.SIGKILL)
    compare.report([TAG])
