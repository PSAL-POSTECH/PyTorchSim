"""Join the cases, the simulated cycles and a TPU reference into one table.

Reads each log's `Total execution cycles` line and, optionally, a measured TPU
csv (workload,cycles[,dtype]).  One row per (case, width).
"""

import argparse
import collections
import csv
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cases import load  # noqa: E402
from roofline import cost, regime  # noqa: E402

LOG_DIR = os.path.join(HERE, "logs")


#: A run that never compiled still prints a cycle line, so the line alone is not
#: evidence.  These mark the log as a failure; scoring it would report a crash as
#: a fast kernel (a compile abort reads as 1 cycle, i.e. -100% against any ref).
FAILED_MARKS = ("InductorError", "CompilationError", "Traceback (most recent call last)")


def cycles_from_log(name, dtype):
    """The simulated cycle count for one case at one width, or None if it did not run."""
    suffix = "" if dtype == "float32" else f".{dtype}"
    path = os.path.join(LOG_DIR, f"{name}{suffix}.log")
    if not os.path.exists(path):
        return None
    text = open(path, errors="ignore").read()
    if any(mark in text for mark in FAILED_MARKS):
        return None
    m = re.search(r"Total execution cycles:\s*([0-9]+)", text)
    return int(m.group(1)) if m else None


def load_reference(paths):
    """(workload, dtype) -> reference cycles.

    A csv with no dtype column (the old TPUv3 baseline) files under "any", which
    is also a warning: it was measured at a width nobody wrote down.
    """
    ref = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("workload") or row.get("Workload") or "").lstrip("﻿")
                val = row.get("cycles") or row.get("TPUv3") or ""
                if key and val:
                    ref[(key, row.get("dtype", "any"))] = float(val)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="")
    ap.add_argument("--origin", default="")
    ap.add_argument("--dtype", default="float32,float16")
    ap.add_argument("--machine", default="v6e", choices=["v3", "v6e"],
                    help="which machine's ridge labels the regime column")
    #: ref_v6e.csv is the real v6e device time, written by gen_v6e_cases.py and keyed
    #: at float16 because the measurements are bfloat16 -- same bytes, same timing.
    #: The TPUv3 baselines it replaced were a different machine at an unrecorded width.
    ap.add_argument("--reference", nargs="*", default=[
        os.path.join(HERE, "ref_v6e.csv")])
    ap.add_argument("--out", default=os.path.join(HERE, "timing_summary.csv"))
    args = ap.parse_args()

    ref = load_reference(args.reference)
    cases = load([o for o in args.op.split(",") if o] or None,
                 origin=set(o for o in args.origin.split(",") if o))
    rows, errs = [], []
    for c in cases:
        macs, byts = cost(c["op"], c["size"])
        for dtype in (d for d in args.dtype.split(",") if d):
            width = byts if dtype == "float32" else byts // 2
            sim = cycles_from_log(c["name"], dtype)
            tpu = ref.get((c["name"], dtype)) or ref.get((c["name"], "any"))
            err = ((sim - tpu) / tpu * 100.0) if (sim and tpu) else None
            if err is not None:
                errs.append(abs(err))
            rows.append({"workload": c["name"], "op": c["op"], "dtype": dtype,
                         "origin": c["origin"], "census_match": c["census"],
                         "regime": regime(macs, width, args.machine),
                         "sim_cycles": sim or "", "ref_cycles": tpu or "",
                         "error_pct": round(err, 1) if err is not None else ""})

    w = max((len(r["workload"]) for r in rows), default=10)
    print(f"{'workload':<{w}} {'dtype':>9} {'origin':>7} {'census':>7} "
          f"{args.machine:>9} {'sim':>12} {'ref':>12} {'err%':>8}")
    print("-" * (w + 70))
    for r in rows:
        print(f"{r['workload']:<{w}} {r['dtype']:>9} {r['origin']:>7} "
              f"{r['census_match']:>7} {r['regime']:>9} {str(r['sim_cycles']):>12} "
              f"{str(r['ref_cycles']):>12} {str(r['error_pct']):>8}")

    done = sum(1 for r in rows if r["sim_cycles"] != "")
    by_op = collections.Counter(r["op"] for r in rows if r["sim_cycles"] != "")
    print(f"\n{done}/{len(rows)} simulated across {len(by_op)} ops, {len(errs)} comparable"
          + (f", MAE {statistics.mean(errs):.1f}% (median {statistics.median(errs):.1f}%)"
             if errs else ""))
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
