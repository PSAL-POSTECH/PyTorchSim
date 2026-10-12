"""The ops the pipeline knows. Adding one = an entry here + its program in sim_op.py.

    keys     the shape columns of the TPU table, in order (tpu-perf-archive tools/ops.yaml)
    size     shape -> work, the order shapes are simulated in (smallest first)
    carry    TPU-table columns copied into compare.csv
    buckets  compare row -> the summary lines it belongs to, besides "all" and "TPU status ok",
             as (rank, name); the summary prints them by rank
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_URL = "git@github.com:PSAL-POSTECH/tpu-perf-archive.git"


def _gemm_buckets(r):
    m = [(lo, f"M {lo}-{hi}") for lo, hi in ((1, 1), (32, 128), (256, 1024), (2048, 8192)) if lo <= r["M"] <= hi]
    return m + [(1 << 20 if r["bound"] == "memory" else 1 << 21, f"{r['bound']}-bound")]


def _softmax_buckets(r):
    e = r["R"] * r["C"]
    size = ((0, "elements < 64K") if e < 1 << 16 else (1, "64K - 1M") if e < 1 << 20 else
            (2, "1M - 16M") if e < 1 << 24 else (3, ">= 16M"))
    return [size, (4 + r["C"], f"C = {r['C']}")]


OPS = {
    "gemm": dict(keys=("M", "K", "N"), size=lambda s: s[0] * s[1] * s[2],
                 carry=("path", "bound"), buckets=_gemm_buckets),
    "softmax": dict(keys=("R", "C"), size=lambda s: s[0] * s[1],
                    carry=("gbps", "source"), buckets=_softmax_buckets),
}


def table_path(op):
    """The op's TPU table: in $TPU_PERF_ARCHIVE if set, else in a clone next to this file, made on first use."""
    root = os.environ.get("TPU_PERF_ARCHIVE") or os.path.join(HERE, "tpu-perf-archive")
    if not os.path.isdir(root):
        print(f"cloning {ARCHIVE_URL} -> {root}", flush=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", ARCHIVE_URL, root], check=True)
    return os.path.join(root, "data", "v6e", op, "torchxla.csv")
