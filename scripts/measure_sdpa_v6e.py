"""Performance measurement harness for flash-SDPA on the TPU v6e schema.

Runs the same shape sweep as tests/ops/attention/test_sdpa.py but with Spike
(functional validation) OFF -- only Gem5 (compute-latency tables) + TOGSim
(cycle-accurate) run. For each case it parses the TOGSim log once and writes a
fully enriched row (cycles, SA/VPU utilization, DRAM bandwidth AND the per-case
DMA metrics) directly, so no separate analyze/enrich pass over a stale
intermediate CSV is needed -- the CSV this emits is already the enriched schema.

Usage:
    export TORCHSIM_DIR=/workspace/PyTorchSim
    export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_measure.yml
    python scripts/measure_sdpa_v6e.py [csv_path]
"""
import os, re, sys, csv, glob, time

BASE = os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tests/ops/attention"))
import torch
import test_sdpa as t

LOG_DIR = os.path.join(BASE, "togsim_results")
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "sdpa_results", "v6e_gem5_measure_enriched.csv")
os.makedirs(os.path.dirname(CSV), exist_ok=True)

PEAK_BW = 1638.4  # GB/s (HBM3 32ch), for dram_util_pct

# Enriched schema, emitted directly (superset of the old raw schema). The
# trailing `log` column is kept for traceability; downstream readers use named
# columns so the extra field is harmless.
FIELDS = ["mode", "gqa", "B", "Hq", "Hkv", "S", "D", "total_cycles",
          "sa0_util", "sa1_util", "vpu_util", "dram_bw_gbs", "dram_util_pct",
          "dma_active", "dma_idle", "dram_responses", "read_latency", "log"]

_num = lambda m: float(m.group(1)) if m else None

def parse_latest_log(before_ts):
    logs = [p for p in glob.glob(os.path.join(LOG_DIR, "*.log")) if os.path.getmtime(p) >= before_ts]
    if not logs:
        return {}
    path = max(logs, key=os.path.getmtime)
    txt = open(path, "r", errors="ignore").read()
    sa = re.findall(r"Systolic array \[(\d)\] utilization\(%\):\s*([\d.]+)", txt)
    sa_util = {int(i): float(v) for i, v in sa}
    # Final cumulative Core DMA summary (last occurrence): active/idle cycles,
    # DRAM bandwidth and response count in one line, plus avg read latency.
    dma = re.findall(r"DMA active_cycles:\s*(\d+),\s*DMA idle_cycles:\s*(\d+),\s*DRAM BW:\s*([\d.]+) GB/s \((\d+) responses\)", txt)
    rl = re.findall(r"avg_read_latency:\s*([\d.]+)", txt)
    dma_active = dma_idle = dram_bw = dram_responses = read_latency = None
    if dma:
        a, i, bw, resp = dma[-1]
        dma_active, dma_idle, dram_bw, dram_responses = int(a), int(i), float(bw), int(resp)
        read_latency = float(rl[-1]) if rl else None
    if dram_bw is None:  # fall back to the standalone BW line if the combined one is absent
        dram_bw = _num(re.search(r"DMA active_cycles.*?DRAM BW:\s*([\d.]+) GB/s", txt))
    dram_util_pct = round(100.0 * dram_bw / PEAK_BW, 2) if dram_bw is not None else None
    return {
        "total_cycles": int(m.group(1)) if (m := re.search(r"Total execution cycles:\s*(\d+)", txt)) else None,
        "sa0_util": sa_util.get(0),
        "sa1_util": sa_util.get(1),
        "vpu_util": _num(re.search(r"Vector unit utilization\(%\):\s*([\d.]+)", txt)),
        "dram_bw_gbs": dram_bw,
        "dram_util_pct": dram_util_pct,
        "dma_active": dma_active,
        "dma_idle": dma_idle,
        "dram_responses": dram_responses,
        "read_latency": read_latency,
        "log": os.path.basename(path),
    }

def run_case(writer, fh, mode, gqa, B, Hq, Hkv, S, D):
    L = 1 if mode == "decode" else S
    q = torch.rand(B, Hq,  L, D, dtype=torch.float16)
    k = torch.rand(B, Hkv, S, D, dtype=torch.float16)
    v = torch.rand(B, Hkv, S, D, dtype=torch.float16)
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False)
    if gqa:
        kwargs["enable_gqa"] = True
    t.clear_caches()
    ts = time.time()
    try:
        t._run_sdpa(t.device, q, k, v, **kwargs).cpu()
        stats = parse_latest_log(ts)
        status = "ok"
    except Exception as e:
        stats = {}
        status = f"ERROR:{type(e).__name__}"
    row = {"mode": mode, "gqa": int(gqa), "B": B, "Hq": Hq, "Hkv": Hkv, "S": S, "D": D, **stats}
    writer.writerow(row); fh.flush()
    print(f"[{status}] {mode} gqa={int(gqa)} B{B} Hq{Hq} Hkv{Hkv} S{S} D{D} -> "
          f"cycles={stats.get('total_cycles')} sa0={stats.get('sa0_util')} "
          f"sa1={stats.get('sa1_util')} vpu={stats.get('vpu_util')} bw={stats.get('dram_bw_gbs')}",
          flush=True)

def main():
    torch.manual_seed(0)
    fh = open(CSV, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    n = 0
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        # MHA prefill + decode. Prefill is O(S^2) so cap it; decode (L=1) is cheap
        # and stays on the full S range for long-context KV-cache scaling.
        PREFILL_S_CAP = 2048
        for mode in ("prefill", "decode"):
            seqs = t.SEQ_LIST if mode == "decode" else [s for s in t.SEQ_LIST if s <= PREFILL_S_CAP]
            for B in t.BATCH_LIST:
                for H in t.HEAD_LIST:
                    for S in seqs:
                        for D in t.HEAD_DIM_LIST:
                            run_case(writer, fh, mode, False, B, H, H, S, D); n += 1
        # GQA prefill + decode
        for mode in ("prefill", "decode"):
            for B in t.BATCH_LIST:
                for Hkv, ratio in t.GQA_HEAD_CONFIGS:
                    Hq = ratio * Hkv
                    for S in t.GQA_SEQ_LIST:
                        for D in t.GQA_HEAD_DIM_LIST:
                            run_case(writer, fh, mode, True, B, Hq, Hkv, S, D); n += 1
    fh.close()
    print(f"DONE: {n} cases measured -> {CSV}", flush=True)

if __name__ == "__main__":
    main()
