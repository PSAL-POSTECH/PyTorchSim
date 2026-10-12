"""runs/<tag>/results.csv (sim) x TPU table -> runs/<tag>/compare.csv + a summary on stdout.

    python compare.py TAG [TAG ...]      a tag of --out, or a path to a tag directory

Every op gets the same numbers: sim_over_tpu = sim_ns / tpu_ns per shape, where tpu_ns is the
table's latency_ns (the per-execution median), then median, IQR, min, max and MAPE per bucket.
With several tags the last table puts them side by side.
"""
import argparse
import csv
import json
import os
import statistics as st

from ops import OPS

HERE = os.path.dirname(os.path.abspath(__file__))


def compare(tag_dir):
    meta = json.load(open(os.path.join(tag_dir, "meta.json")))
    op = OPS[meta["op"]]
    key = lambda r: tuple(int(r[k]) for k in op["keys"])  # noqa: E731
    tpu = {key(r): r for r in csv.DictReader(open(meta["table"]))}
    results = os.path.join(tag_dir, "results.csv")
    sim = {key(r): r for r in csv.DictReader(open(results))} if os.path.exists(results) else {}
    rows = []
    for k in sorted(sim, key=lambda s: (op["size"](s), s)):
        t, s = tpu[k], sim[k]
        sim_ns, tpu_ns = int(s["sim_ns"]), int(t["latency_ns"])
        rows.append(dict(zip(op["keys"], k), tpu_ns=tpu_ns, sim_ns=sim_ns, sim_cycles=s["cycles"],
                         sim_over_tpu=round(sim_ns / tpu_ns, 4), tpu_status=t["status"],
                         **{c: t[c] for c in op["carry"]}))
    if rows:
        with open(os.path.join(tag_dir, "compare.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return meta, rows


def stats(rs):
    r = [x["sim_over_tpu"] for x in rs]
    q = st.quantiles(r, n=4) if len(r) > 1 else [r[0]] * 3
    return (f"n={len(r):3}  sim/TPU median {st.median(r):.3f}  IQR {q[0]:.3f}-{q[2]:.3f}  "
            f"min {min(r):.3f}  max {max(r):.3f}  MAPE {100 * st.mean(abs(x - 1) for x in r):.1f}%")


def report(tag_dirs):
    side = []
    for d in tag_dirs:
        meta, rows = compare(d)
        tag = os.path.basename(os.path.normpath(d))
        print(f"\n{tag}: {meta['op']}, {len(rows)} simulated, {meta['torchsim_sha']} / {meta['compile_sha']}"
              f"{', set ' + ' '.join(meta['set']) if meta['set'] else ''}")
        if not rows:
            continue
        ok = [x for x in rows if x["tpu_status"] == "ok"]
        buckets = {}
        for x in rows:
            for b in OPS[meta["op"]]["buckets"](x):
                buckets.setdefault(b, []).append(x)
        for name, rs in [("all", rows), ("TPU status ok", ok)] + [(b[1], buckets[b]) for b in sorted(buckets)]:
            if rs:
                print(f"  {name:18} {stats(rs)}")
        if ok:
            side.append((f"{tag} ({meta['op']})", stats(ok)))
    if len(side) > 1:
        print("\nTPU status ok, side by side")
        w = max(len(n) for n, _ in side)
        for name, s in side:
            print(f"  {name:{w}}  {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"), help="where the tag directories live")
    a = ap.parse_args()
    report([t if os.path.isdir(t) else os.path.join(a.out, t) for t in a.tags])
