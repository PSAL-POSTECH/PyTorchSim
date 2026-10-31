"""Merge a grid sweep's summary.tsv into sim.csv, in microseconds at the target's own clock.

A point is taken only when its run finished (rc 0) and its log carried cycles; every other point keeps the
microseconds that are already in sim.csv, so a sweep that was cut short refines the table instead of emptying it.
"""
import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def read_mhz(config):
    """The core clock the cycles were counted at, read from the machine description, never written here."""
    with open(config) as fh:
        for line in fh:
            m = re.match(r"\s*core_freq_mhz\s*:\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    raise SystemExit(f"{config} has no core_freq_mhz: the clock is the machine's, not this script's")


def read_runs(paths, mhz):
    """{(model, mode, S): microseconds} of the points that ran to the end, newest sweep winning."""
    out = {}
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("rc") == "0" and r.get("cycles") and int(r["cycles"]) > 0:
                    out[(r["name"], r["mode"], int(r["S"]))] = int(r["cycles"]) / mhz
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="summary.tsv files, or directories holding one")
    ap.add_argument("--sim", default=os.path.join(HERE, "sim.csv"))
    ap.add_argument("--config", required=True, help="the TOGSim machine description the sweep ran on")
    args = ap.parse_args()

    paths = []
    for r in args.runs:
        paths += sorted(glob.glob(os.path.join(r, "**", "summary.tsv"), recursive=True)) if os.path.isdir(r) else [r]
    fresh = read_runs(paths, read_mhz(args.config))

    rows = list(csv.DictReader(open(args.sim)))
    taken = 0
    for row in rows:
        k = (row["model"], row["mode"], int(row["S"]))
        if k in fresh:
            row["sim_us"] = f"{fresh[k]:.1f}"
            taken += 1
    known = {(r["model"], r["mode"], int(r["S"])) for r in rows}
    for k in sorted(fresh):
        if k not in known:
            rows.append(dict(model=k[0], mode=k[1], S=k[2], sim_us=f"{fresh[k]:.1f}"))
            taken += 1
    with open(args.sim, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["model", "mode", "S", "sim_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(paths)} sweep file(s), {len(fresh)} finished point(s), {taken} row(s) taken into {args.sim}")


main()
