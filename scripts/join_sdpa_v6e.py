#!/usr/bin/env python3
"""Join real TPU v6e measurements with TOGSim (gem5) simulation results.

Inputs (under sdpa_results/):
  v6e_tpu_measure.csv           - real TPU profiler sweep (real_*)
  v6e_gem5_measure_enriched.csv - TOGSim simulation sweep (sim_*)

Output:
  v6e_validation_join.csv       - inner join on the shape keys, with
                                  sim/real ratio and abs percent error.

Only the shapes present in BOTH sweeps are emitted. Rows are sorted by
abs_pct_err descending (worst simulator mismatch first).
"""
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "sdpa_results")
TPU = os.path.join(RESULTS, "v6e_tpu_measure.csv")
SIM = os.path.join(RESULTS, "v6e_gem5_measure_enriched.csv")
OUT = os.path.join(RESULTS, "v6e_validation_join.csv")

# Shape identity shared by both sweeps.
KEYS = ["mode", "gqa", "B", "Hq", "Hkv", "S", "D"]

OUT_COLS = KEYS + [
    "real_cycles", "sim_cycles", "ratio_sim_real", "abs_pct_err",
    "real_mxu", "sim_sa_max", "real_hbm_gbs", "sim_dram_gbs",
    "real_vpu", "sim_vpu",
]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(row[k] for k in KEYS)


def fnum(s):
    """Parse a float, treating blank cells as missing (None)."""
    s = (s or "").strip()
    if not s:
        return None
    return float(s)


def main():
    tpu = load(TPU)
    sim = {key(r): r for r in load(SIM)}

    rows = []
    for r in tpu:
        s = sim.get(key(r))
        if s is None:
            continue  # inner join: shape must exist in the sim sweep too

        real_cycles = fnum(r["total_cycles"])
        sim_cycles = fnum(s["total_cycles"])
        if real_cycles is None or sim_cycles is None or real_cycles == 0:
            continue

        ratio = sim_cycles / real_cycles
        abs_pct_err = abs(sim_cycles - real_cycles) / real_cycles * 100.0

        sa0 = fnum(s.get("sa0_util"))
        sa1 = fnum(s.get("sa1_util"))
        sa_vals = [v for v in (sa0, sa1) if v is not None]
        sim_sa_max = max(sa_vals) if sa_vals else None

        def g(v):
            return "" if v is None else v

        rows.append({
            **{k: r[k] for k in KEYS},
            "real_cycles": int(real_cycles),
            "sim_cycles": int(sim_cycles),
            "ratio_sim_real": round(ratio, 3),
            "abs_pct_err": round(abs_pct_err, 1),
            "real_mxu": g(fnum(r.get("mxu_util"))),
            "sim_sa_max": g(sim_sa_max),
            "real_hbm_gbs": g(fnum(r.get("hbm_bw_gbs"))),
            "sim_dram_gbs": g(fnum(s.get("dram_bw_gbs"))),
            "real_vpu": g(fnum(r.get("vpu_util"))),
            "sim_vpu": g(fnum(s.get("vpu_util"))),
        })

    rows.sort(key=lambda x: x["abs_pct_err"], reverse=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"[join] {len(rows)} shapes matched -> {OUT}")


if __name__ == "__main__":
    main()
