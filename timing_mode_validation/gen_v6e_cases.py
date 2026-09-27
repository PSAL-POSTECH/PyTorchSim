"""Derive cases and their reference from the real-v6e measurements.

The measurement tree is the source of truth: a case exists because silicon ran that
exact shape, and its reference is that run's device time.  One measured program ->
one op file, because pairing two different programs would price two things as one.

  ops/gemm.py           qkv, out_proj, ffn_g1, ffn_g2, lm_head   (one GEMM each)
  ops/ffn_gated.py      decoder ffn: fused [h, 2*inter] gate/up, silu*up, down
  ops/ffn_erf.py        encoder ffn_erf: two GEMMs with bias, erf GELU between
  ops/attn_enc_full.py  encoder attn: non-causal, materialised scores (XLA's path)
  ops/attn_dec_gqa.py   decoder attn: the OPERATION ragged_paged_attention performs

Rewrites those five files and `ref_v6e.csv`; nothing else generates any of them.
"""

import argparse
import csv
import glob
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE = "/workspace/tpuv6e_validation"

#: v6e config clock -- cycles are the simulator's unit, so its own clock converts.
#: configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml:core_freq_mhz
CORE_FREQ_MHZ = 3502

#: Measurements are bfloat16.  The harness pairs them with its float16 runs: the
#: two differ in exponent/mantissa split, not in bytes, and timing reads bytes.
DTYPE = "float16"

#: M values kept per weight shape.  The measured grids run to 87 points, which is
#: a sweep, not a case list; this ladder crosses the knees the array cares about
#: (1 -> the 256-row array height -> weight-streaming saturation) and drops the
#: interpolation filler between them.  The largest measured M is always kept.
LADDER = [1, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

#: Decoder attention is a (q, kv) grid of 110 points per model, which is a sweep.
#: These cross decode (q=1), chunked prefill and full prefill against a cache that
#: runs from one page to the 128k the serving path allows.
Q_LADDER = [1, 32, 128, 512, 2048]
KV_LADDER = [128, 512, 2048, 8192, 32768, 131072]

#: We materialise the [H, q, kv] scores that silicon's paged kernel never builds, so
#: a case whose score tensor alone passes this is not a timing question any more.
MAX_SCORE_BYTES = 2 << 30


def models(tree):
    """MODELS from the sweep harness -- the only place the dims are written down."""
    src = open(os.path.join(tree, "code/harness/tpu_lut_sweep_v2.py")).read()
    ns = {}
    exec(re.search(r"^MODELS = \{.*?^\}", src, re.S | re.M).group(0), ns)
    return ns["MODELS"]


def weight_shape(m, op):
    """(K, N) of the one weight this op multiplies, or None if it is not one GEMM."""
    h, inter, enc = m["h"], m["inter"], m["type"] == "enc"
    return {"qkv": (h, (m["hq"] + 2 * m["hkv"]) * m["d"]),
            "out_proj": (m["hq"] * m["d"], h),
            "ffn_g1": (h, inter if enc else 2 * inter),
            "ffn_g2": (inter, h),
            "lm_head": (h, m["vocab"])}.get(op)


def collect(tree):
    """(M, K, N) -> [(model, op, latency_ns)] over every measured linear GEMM."""
    MODELS = models(tree)
    out = {}
    for path in sorted(glob.glob(os.path.join(tree, "data/linear_gemm/*.csv"))):
        model, op = os.path.basename(path)[:-4].split("__")
        shape = weight_shape(MODELS[model], op)
        if shape is None:          # ffn / ffn_gemm_only are two GEMMs in one program
            continue
        for row in csv.DictReader(open(path)):
            key = (int(row["tokens"]),) + shape
            out.setdefault(key, []).append((model, op, float(row["latency_ns"])))
    return out


def select(points):
    """The ladder rows, plus the largest M measured for each weight shape."""
    by_shape = {}
    for M, K, N in points:
        by_shape.setdefault((K, N), []).append(M)
    keep = set()
    for shape, ms in by_shape.items():
        wanted = {m for m in ms if m in LADDER} | {max(ms)}
        keep |= {(m,) + shape for m in wanted}
    return sorted(keep, key=lambda k: (k[1], k[2], k[0]))


#: Which models the two ffn programs belong to -- the harness writes a decoder's ffn
#: and its ffn_erf from the same source, so on decoders those two tables are one
#: program measured twice (checked: median ratio 0.996-1.012).
def ffn_points(tree, kind):
    """(S, h, inter) -> [(model, latency_ns)] for the decoder or encoder ffn."""
    MODELS = models(tree)
    sub, out = ("linear_gemm/%s__ffn.csv" if kind == "dec"
                else "attn_ffn/%s__ffn_erf.csv"), {}
    for model, m in MODELS.items():
        if (m["type"] == "dec") != (kind == "dec"):
            continue
        path = os.path.join(tree, "data", sub % model)
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path)):
            key = (int(row["tokens"]), m["h"], m["inter"])
            out.setdefault(key, []).append((model, float(row["latency_ns"])))
    return out


def attn_points(tree, kind):
    """The measured attention grid, keyed by the op's own size tuple."""
    MODELS = models(tree)
    out = {}
    for model, m in MODELS.items():
        if (m["type"] == "dec") != (kind == "dec"):
            continue
        path = os.path.join(tree, f"data/attn_ffn/{model}__attn.csv")
        if not os.path.exists(path):
            continue
        if kind == "dec" and m["d"] != 128:
            #: d=64 is zero-padded to 128 by the ragged kernel, so that row's time is
            #: an upper bound for twice the KV bytes -- not this model's own cost.
            print(f"  건너뜀 {model}: d={m['d']} 가 128 로 패딩돼 측정됐다(상한값)")
            continue
        for row in csv.DictReader(open(path)):
            q, kv = int(row["query"]), int(row["kv"])
            ns = float(row["latency_ns"])
            if kind == "enc":
                if q != kv:
                    continue
                key = (m["hq"], q, m["d"])
            else:
                if q not in Q_LADDER or kv not in KV_LADDER or q > kv:
                    continue
                if m["hq"] * q * kv * 4 > MAX_SCORE_BYTES:
                    continue
                key = (m["hq"], m["hkv"], q, kv, m["d"])
            out.setdefault(key, []).append((model, ns))
    return out


def merge(*dicts):
    """Points from several tables under one key -- a shape two tables both measured
    keeps both units, so rows() takes their median and says how far apart they were."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def attn_batch_points(tree, kind):
    """The batch tables, folded onto the same op. A sequence and a head are both an
    independent score matrix, so `seqs` multiplies the head count and no new program
    is needed -- for GQA the fold keeps each sequence's kv heads adjacent, which is the
    layout repeat_interleave produces."""
    MODELS = models(tree)
    out = {}
    for model, m in MODELS.items():
        if (m["type"] == "dec") != (kind == "dec"):
            continue
        path = os.path.join(tree, f"data/attn_ffn/{model}__attn/attn_batch_tp1.csv")
        if not os.path.exists(path):
            continue
        if kind == "dec" and m["d"] != 128:
            continue                      # d=64 was padded to 128 when measured
        for row in csv.DictReader(open(path)):
            seqs, ns = int(row["seqs"]), float(row["latency_ns"])
            if kind == "enc":
                S = int(row["seq"])
                key = (m["hq"] * seqs, S, m["d"])
                if m["hq"] * seqs * S * S * 4 > MAX_SCORE_BYTES:
                    continue
            else:
                q, kv = int(row["query"]), int(row["kv"])
                if q > kv:
                    continue
                key = (m["hq"] * seqs, m["hkv"] * seqs, q, kv, m["d"])
                if m["hq"] * seqs * q * kv * 4 > MAX_SCORE_BYTES:
                    continue
            out.setdefault(key, []).append((f"{model}/batch{seqs}", ns))
    return out


def rows(points, name_of, keep=None):
    """(cases, ref rows) for one program: the median over units that share a shape."""
    cases, ref = [], []
    for key in sorted(points):
        if keep and not keep(key):
            continue
        units = points[key]
        ns = statistics.median(u[1] for u in units)
        who = ",".join(sorted({u[0] for u in units}))
        spread = ((max(u[1] for u in units) - min(u[1] for u in units)) / ns
                  if len(units) > 1 else 0.0)
        note = (f"같은 shape 을 잰 단위 {len(units)}개, 편차 {spread * 100:.1f}%"
                if len(units) > 1 else "")
        name = name_of(key)
        cases.append((name, key, "v6e", "exact", who, note))
        ref.append({"workload": name, "dtype": DTYPE,
                    "cycles": round(ns * 1e-9 * CORE_FREQ_MHZ * 1e6),
                    "latency_ns": round(ns), "source": who, "n_units": len(units)})
    return cases, ref


def write_op(path, params, doc, cases):
    """One generated case file. The only writer of any ops/*.py under this script."""
    with open(os.path.join(HERE, path), "w") as f:
        f.write(f'"""{doc}\n\nGENERATED by gen_v6e_cases.py -- do not hand-edit.\n'
                'Every case is a shape silicon ran; ref_v6e.csv carries its device time.\n'
                f'"""\n\nPARAMS = "{params}"\n\n'
                '#: (name, size, origin, census, source, note).\n'
                '#: origin  v6e (a real TPU v6e measured this exact shape)\n'
                '#: census  exact (the measurement ran it; nothing here is interpolated)\n'
                'CASES = [\n')
        for name, size, origin, census, source, note in cases:
            f.write(f'    ("{name}", {tuple(size)}, "{origin}", "{census}", '
                    f'"{source}", "{note}"),\n')
        f.write(']\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=DEFAULT_TREE, help="tpuv6e_validation root")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = ap.parse_args()

    gemm_pts = collect(args.tree)
    plan = [
        ("ops/gemm.py", "M, K, N",
         "gemm -- one GEMM: qkv, out_proj, ffn_g1, ffn_g2, lm_head.",
         *rows({k: [(f"{a}/{b}", ns) for a, b, ns in v] for k, v in gemm_pts.items()},
               lambda k: "gemm_%dx%dx%d" % k, keep=set(select(gemm_pts)).__contains__)),
        ("ops/ffn_gated.py", "S, hidden, interm",
         "ffn_gated -- the decoder ffn: one [h, 2*inter] gate/up GEMM, silu*up, down.",
         *rows(ffn_points(args.tree, "dec"), lambda k: "ffn_gated_%dx%dx%d" % k,
               keep=lambda k: k[0] in LADDER)),
        ("ops/ffn_erf.py", "S, hidden, interm",
         "ffn_erf -- the encoder ffn that represents serving: bias GEMMs, erf GELU.",
         *rows(ffn_points(args.tree, "enc"), lambda k: "ffn_erf_%dx%dx%d" % k,
               keep=lambda k: k[0] in LADDER)),
        ("ops/attn_enc_full.py", "Hq, S, D",
         "attn_enc_full -- encoder attention: non-causal, materialised scores.",
         *rows(merge(attn_points(args.tree, "enc"), attn_batch_points(args.tree, "enc")),
               lambda k: "attn_enc_%dx%dx%d" % k)),
        ("ops/attn_dec_gqa.py", "Hq, Hkv, q_len, kv_len, D",
         "attn_dec_gqa -- the operation ragged_paged_attention performs, materialised.",
         *rows(merge(attn_points(args.tree, "dec"), attn_batch_points(args.tree, "dec")),
               lambda k: "attn_dec_%dx%dx%dx%dx%d" % k)),
    ]
    ref = []
    for path, params, doc, cases, refs in plan:
        print(f"{path:24} {len(cases):4} cases")
        ref += refs
        if not args.dry_run:
            write_op(path, params, doc, cases)
    if args.dry_run:
        print(f"ref_v6e.csv  {len(ref)} rows (미작성)")
        return
    with open(os.path.join(HERE, "ref_v6e.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref[0]))
        w.writeheader()
        w.writerows(ref)
    print(f"ref_v6e.csv  {len(ref)} rows  ({DTYPE}, {CORE_FREQ_MHZ} MHz)")


if __name__ == "__main__":
    main()
