#!/usr/bin/env python3
"""Run the existing test suite through the Triton codegen route.

Produces a gate (triton_route_passing.txt), a report bucketed by cause and
stage, and per-failure artifacts for reporting upstream.

  python scripts/ci/triton_route_sweep.py                  # the allowlist, gating
  python scripts/ci/triton_route_sweep.py --all            # every test, reports
  python scripts/ci/triton_route_sweep.py --all --artifacts triton-failures
  python scripts/ci/triton_route_sweep.py --all --order-from results.json

An entry is a test file, or a test file and its arguments when the allowlist runs one file as
several entries (`tests/models/Gemma/test_gemma.py --preset 7b`): each is its own process, so a
long file does not hold the last slot alone.
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim import contract  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
PASSING = os.path.join(HERE, "triton_route_passing.txt")

#: Each step and the IR file it writes last. A kernel stopped in the first step whose file is missing.
STAGES = [
    ("01-ttir.mlir",        "1 triton-compile: frontend"),
    ("07-regularized.mlir", "1 triton-compile: TTGIR -> linalg"),
    ("08-transformed.mlir", "2 torchsim-compile: transform"),
    ("10-lowered.mlir",     "2 torchsim-compile: lower (DMA, lanes, spad)"),
]
#: The steps after those, finished when the manifest (kernel.json) names their output.
OUTPUTS = [("elf", "2 torchsim-compile: binary"), ("tog_so", "TOG")]

#: First match wins. Each bucket names the layer that owns the fix.
BUCKETS = [
    ("missing_dep",    r"ModuleNotFoundError|No module named"),
    ("device_op",      r"\w+_overrideable not implemented|not implemented\. .*privateuse"),
    ("triton_helpers", r"triton_helpers"),
    ("wrapper_gap",    r"'TritonNPUWrapperCodegen' object has no attribute"),
    ("spec_incomplete", r"SpecIncomplete"),
    ("compiler_stage",     r"CompilerError|the compiler pipeline failed|triton-shared-opt|"
                       r"\[stage\d\]|\[(transform|lower|binary)\]|failed to legalize"),
    ("reduction",      r"lane-aware|linalg\.reduce|no reduction path"),
    ("dynamic_shape",  r"ShapeMismatch|dynamic shape|size_hint returned None"),
    ("matmul_timing",  r"vcix\.iv|sf\.vc\.|no compute node"),
    ("togsim",         r"TOGSim|tog\.so|SIGSEGV|Signals\.SIG"),
    ("wrong_values",   r"VALUES WRONG|allclose|Test Failed"),
    ("timeout",        r"^__timeout__$"),
]

#: Lines torch prints alongside an error that are not the error.
NOISE = re.compile(
    r"TORCHDYNAMO_VERBOSE|torch\._dynamo|You can suppress this|set TORCH_LOGS|"
    r"^During handling|^The above exception|for more information|^\s*\^+\s*$")


def discover(allow):
    """Every test file, each one the allowlist runs as entries with arguments replaced by those."""
    split = {}
    for entry in allow:
        file, *argv = shlex.split(entry)
        if argv:
            split.setdefault(file, []).append(entry)
    out = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "tests")):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                path = os.path.relpath(os.path.join(dirpath, f), ROOT)
                out.extend(split.get(path, [path]))
    return sorted(out)


def _slug(entry):
    """A directory name for an entry."""
    return re.sub(r"[^\w.-]+", "_", entry.removesuffix(".py"))


def load_allowlist():
    if not os.path.isfile(PASSING):
        return []
    with open(PASSING) as f:
        return [l.strip() for l in f
                if l.strip() and not l.lstrip().startswith("#")]


def longest_first(tests, path):
    """Order by a previous run's results.json: longest first, a test it has not
    seen before those, ties by path. Path order when it cannot be read."""
    try:
        with open(path) as f:
            prev = {r["test"]: float(r["seconds"]) for r in json.load(f)}
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"--order-from {path}: {e!r}; keeping path order")
        return tests
    known = sum(t in prev for t in tests)
    print(f"--order-from {path}: longest first, {known} of {len(tests)} tests "
          f"have a previous duration")
    return sorted(tests, key=lambda t: (-prev.get(t, float("inf")), t))


def classify(output, timed_out):
    if timed_out:
        return "timeout"
    for name, pat in BUCKETS:
        if re.search(pat, output, re.I | re.M):
            return name
    return "other"


def first_error(output):
    """The exception line, skipping torch's boilerplate around it."""
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if NOISE.search(l):
            continue
        if re.match(r"^\w*(Error|Exception|Failure)\b.*:", l) or \
           re.match(r"^(assert|AssertionError)", l):
            return l[:200]
    for l in reversed(lines):
        if not NOISE.search(l):
            return l[:200]
    return ""


def stopped_in(workdir, timing):
    """The step the kernel in `workdir` did not finish, or None when it finished all the run asks for.
    The compiler starts stage.log, so without one the kernel never reached it."""
    if not os.path.isfile(os.path.join(workdir, "stage.log")):
        return "0 kernel generated, not accepted"
    for fname, label in STAGES:
        if not os.path.isfile(os.path.join(workdir, fname)):
            return label
    for kind, label in OUTPUTS[:None if timing else 1]:
        try:
            done = contract.artifact(workdir, kind)
        except contract.KernelObjectMismatch:
            done = None
        if done is None:
            return label
    return None


def failed_kernel(workdirs, timing, timed_out):
    """(stage, workdir) of a failed test: the kernel whose compile did not finish, the last one
    touched if several did not. No workdir when every kernel finished: the failure came after them.
    """
    unfinished = sorted((wd for wd in workdirs if stopped_in(wd, timing)),
                        key=os.path.getmtime)
    wd = unfinished[-1] if unfinished else None
    if timed_out:
        return (f"timed out in {stopped_in(wd, timing)}" if wd else "timed out"), wd
    if wd:
        return stopped_in(wd, timing), wd
    return ("3 compiled, failed after the compiler" if workdirs
            else "0 nothing emitted"), None


def collect(test, dump_dir, out_root, output, bucket, stage, workdir):
    """One directory per failing test: the kernel, the last IR, the error."""
    dest = os.path.join(out_root, _slug(test))
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "error.txt"), "w") as f:
        f.write(f"test:   {test}\nbucket: {bucket}\nstage:  {stage}\n\n")
        f.write("\n".join(output.strip().splitlines()[-60:]))
    if workdir:
        # The kernel to hand over, and the IR saying where it stopped.
        for name in ("kernel.py", "stage.log", *(s[0] for s in STAGES)):
            src = os.path.join(workdir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, name))
    return dest


def run_one(test, timeout, artifacts, scratch):
    # Private per test: a shared dump lets one test's cached kernel answer for
    # another's.
    dump = os.path.join(scratch, _slug(test))
    shutil.rmtree(dump, ignore_errors=True)
    os.makedirs(dump, exist_ok=True)
    env = dict(os.environ, TORCHSIM_DUMP_PATH=dump)
    # What this sweep checks is values, so it does not pay for cycles (mobilenet_v2 does not
    # finish inside 1800s with timing on). A caller that wants cycles sets TORCHSIM_TIMING_MODE.
    env.setdefault("TORCHSIM_TIMING_MODE", "0")

    t0, timed_out = time.time(), False
    try:
        p = subprocess.run([sys.executable, *shlex.split(test)], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        pre = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out, code, timed_out = pre + "\n__timeout__", 124, True

    ok = code == 0
    workdirs = glob.glob(os.path.join(dump, "triton_*"))
    stage, workdir = (None, None) if ok else failed_kernel(
        workdirs, int(env["TORCHSIM_TIMING_MODE"]), timed_out)
    r = {"test": test, "ok": ok, "returncode": code,
         "seconds": round(time.time() - t0, 1),
         "bucket": None if ok else classify(out, timed_out),
         "stage": stage,
         # No kernel emitted = the route was never used (CPU-only, eager
         # fallback, extern call), so it is not coverage.
         "exercised": bool(workdirs),
         "error": "" if ok else first_error(out)}
    if not ok and artifacts:
        r["artifacts"] = os.path.relpath(
            collect(test, dump, artifacts, out, r["bucket"], stage, workdir), ROOT)
    shutil.rmtree(dump, ignore_errors=True)
    return r


def write_markdown(results, path):
    """The report a human reads: counts by cause, then every failure."""
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    by = {}
    for r in failed:
        by.setdefault(r["bucket"], []).append(r)

    real = [r for r in passed if r["exercised"]]
    L = ["# Triton route coverage", "",
         f"**{len(real)}/{len(results)} pass through the Triton route.** "
         f"({len(passed)-len(real)} more pass without exercising it -- CPU-only, "
         f"eager fallback, or a path that bypasses Inductor.)", ""]
    if failed:
        L += ["| cause | count | owner |", "|---|---|---|"]
        OWNER = {
            "device_op": "device -- op not registered for npu",
            "triton_helpers": "pytorchsim/codegen -- needs a vendored copy",
            "wrapper_gap": "pytorchsim/codegen -- TritonNPUWrapperCodegen incomplete",
            "spec_incomplete": "pytorchsim/codegen -- kernel_metadata cannot describe it",
            "compiler_stage": "the compiler lowering passes",
            "reduction": "the compiler -- no lane-aware reduction",
            "dynamic_shape": "pytorchsim/codegen -- shape-specialised launch",
            "matmul_timing": "build_tog -- compute node lookup",
            "togsim": "TOGSim / the TOG",
            "wrong_values": "numerics -- investigate",
            "missing_dep": "test environment (present in the CI image)",
            "timeout": "too slow, or hung",
            "other": "unclassified",
        }
        for b, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            L.append(f"| {b} | {len(rs)} | {OWNER.get(b, '')} |")
        L += ["", "## Failures", ""]
        for b, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            L.append(f"### {b} ({len(rs)})")
            L.append("")
            for r in sorted(rs, key=lambda r: r["test"]):
                L.append(f"- `{r['test']}` — **{r['stage']}**")
                if r["error"]:
                    L.append(f"  - `{r['error'][:160]}`")
                if r.get("artifacts"):
                    L.append(f"  - artifacts: `{r['artifacts']}`")
            L.append("")
    if real:
        L += ["## Passing through the route", ""]
        L += [f"- `{r['test']}`" for r in real] + [""]
    other = [r for r in passed if not r["exercised"]]
    if other:
        L += ["## Passing without exercising the route", ""]
        L += [f"- `{r['test']}`" for r in other] + [""]
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="run every test, not just the passing allowlist")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--allowlist-timeout", type=int, metavar="SECONDS",
                    help="the timeout of a test on the allowlist, when it is "
                         "to differ from --timeout")
    ap.add_argument("-j", "--jobs", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                    help="tests in flight at once; each may itself use several "
                         "cores (gem5, TOGSim), so this is half the box by "
                         "default")
    ap.add_argument("--order-from", metavar="RESULTS_JSON",
                    help="a previous run's --json: start the tests that took "
                         "longest first, so the last one to finish is not one "
                         "that was started last")
    ap.add_argument("--json", help="write the full result list here")
    ap.add_argument("--artifacts", metavar="DIR",
                    help="per-failure kernel + stage IR + error, for reporting")
    ap.add_argument("--markdown", help="write the human-readable report here")
    ap.add_argument("--update-allowlist", action="store_true",
                    help="rewrite the allowlist from what passed (use with --all)")
    args = ap.parse_args()

    allow = load_allowlist()
    tests = discover(allow) if args.all else allow
    if not tests:
        print("no tests selected; the allowlist is empty and --all was not given")
        return 1
    if args.order_from:
        tests = longest_first(tests, args.order_from)

    scratch = os.path.join(ROOT, ".triton_sweep")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    if args.artifacts:
        shutil.rmtree(args.artifacts, ignore_errors=True)
        os.makedirs(args.artifacts, exist_ok=True)

    limit = {t: args.allowlist_timeout or args.timeout for t in allow}
    print(f"Triton route sweep: {len(tests)} tests, {args.jobs} at a time"
          f"{'' if args.all else ' (allowlist)'}\n")
    results, done = [], 0

    def report(r):
        nonlocal done
        done += 1
        mark = ("ok  " if r["exercised"] else "ok- ") if r["ok"] else "FAIL"
        extra = ("" if r["exercised"] else "  (route not exercised)") if r["ok"] \
            else f"  [{r['bucket']}] @{r['stage']}  {r['error'][:70]}"
        print(f"  {done:3d}/{len(tests)}  {mark} {r['seconds']:7.1f}s  "
              f"{r['test']}{extra}", flush=True)

    if args.jobs == 1:
        for t in tests:
            r = run_one(t, limit.get(t, args.timeout), args.artifacts, scratch)
            results.append(r)
            report(r)
    else:
        # Threads: run_one only waits on a subprocess, and dump dir, Inductor
        # cache and TOGSim FIFO are all already per-test.
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(run_one, t, limit.get(t, args.timeout),
                                args.artifacts, scratch): t for t in tests}
            for fut in cf.as_completed(futs):
                r = fut.result()
                results.append(r)
                report(r)
    results.sort(key=lambda r: r["test"])
    shutil.rmtree(scratch, ignore_errors=True)

    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    real = [r for r in passed if r["exercised"]]

    print(f"\n{'='*72}\npassed {len(passed)}/{len(results)}"
          f"  ({len(real)} through the Triton route, "
          f"{len(passed)-len(real)} without exercising it)")
    if failed:
        by = {}
        for r in failed:
            by.setdefault(r["bucket"], []).append(r)
        print("\nfailures by cause:")
        for b, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"  {b:16s} {len(rs):3d}")
        print("\nwhere they stopped:")
        st = {}
        for r in failed:
            st[r["stage"]] = st.get(r["stage"], 0) + 1
        for s, n in sorted(st.items()):
            print(f"  {s:40s} {n:3d}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")
    if args.markdown:
        write_markdown(results, args.markdown)
        print(f"wrote {args.markdown}")
    if args.artifacts and failed:
        print(f"wrote {args.artifacts}/ ({len(failed)} failure dirs)")

    if args.update_allowlist:
        with open(PASSING, "w") as f:
            f.write("# Tests that pass through the Triton codegen route.\n"
                    "# Gated by scripts/ci/triton_route_sweep.py; regenerate with\n"
                    "#   python scripts/ci/triton_route_sweep.py --all "
                    "--update-allowlist\n")
            for r in real:
                f.write(r["test"] + "\n")
        print(f"wrote {PASSING} ({len(real)} tests)")
        return 0

    regressed = [r for r in failed if r["test"] in allow]
    if regressed:
        print(f"\nREGRESSION: {len(regressed)} allowlisted test(s) failed")
        for r in regressed:
            print(f"  {r['test']}  [{r['bucket']}] {r['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
