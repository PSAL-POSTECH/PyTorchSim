"""Compute the aggregates for the v6e SDPA characterization dashboard.

Reads the enriched measurement CSV produced directly by measure_sdpa_v6e.py
(cycles, SA/VPU utilization, DRAM bandwidth AND per-case DMA metrics already
in-file) and emits a compact JSON (data for the HTML report). This script no
longer parses TOGSim logs or writes the enriched CSV -- that is now the
measurement harness's job."""
import csv, os, json, statistics as st

BASE = "/workspace/PyTorchSim"
SRC = os.path.join(BASE, "sdpa_results", "v6e_gem5_measure_enriched.csv")
JOUT = os.path.join(BASE, "sdpa_results", "v6e_report_data.json")
PEAK_BW = 1638.4  # GB/s (HBM3 32ch)

def _int(s):
    s = str(s).strip()
    return int(s) if s not in ("", "None") else None

def _float(s):
    s = str(s).strip()
    return float(s) if s not in ("", "None") else None

# Keep only successfully-measured cases (error rows have a blank total_cycles).
rows = [r for r in csv.DictReader(open(SRC)) if str(r.get("total_cycles", "")).strip()]
for r in rows:
    r["c"] = int(r["total_cycles"]); r["Hq"] = int(r["Hq"]); r["Hkv"] = int(r["Hkv"]); r["B"] = int(r["B"])
    r["S"] = int(r["S"]); r["D"] = int(r["D"])
    r["sa_avg"] = (float(r["sa0_util"]) + float(r["sa1_util"])) / 2
    r["vpu"] = float(r["vpu_util"]); r["bw"] = float(r["dram_bw_gbs"])
    r["dram_util"] = 100.0 * r["bw"] / PEAK_BW
    # DMA metrics come straight from the enriched CSV (no log re-parsing).
    r["dma_active"] = _int(r.get("dma_active")); r["dma_idle"] = _int(r.get("dma_idle"))
    r["dram_responses"] = _int(r.get("dram_responses")); r["read_latency"] = _float(r.get("read_latency"))

def grp(pred):
    g = [r for r in rows if pred(r)]
    c = [r["c"] for r in g]
    if not g:
        return dict(n=0, cmin=None, cmed=None, cmax=None, cmean=None, sa=None, vpu=None, dram=None, reads=None)
    return dict(n=len(g), cmin=min(c), cmed=int(st.median(c)), cmax=max(c), cmean=int(st.mean(c)),
                sa=round(st.mean(r["sa_avg"] for r in g),1), vpu=round(st.mean(r["vpu"] for r in g),1),
                dram=round(st.mean(r["dram_util"] for r in g),1),
                reads=int(st.mean(r["dram_responses"] for r in g if r["dram_responses"] is not None)))

groups = {
  "MHA prefill": grp(lambda r: r["mode"]=="prefill" and r["gqa"]=="0"),
  "MHA decode":  grp(lambda r: r["mode"]=="decode"  and r["gqa"]=="0"),
  "GQA prefill": grp(lambda r: r["mode"]=="prefill" and r["gqa"]=="1"),
  "GQA decode":  grp(lambda r: r["mode"]=="decode"  and r["gqa"]=="1"),
  "ALL":         grp(lambda r: True),
}

# H1 bottleneck classification per case
def bottleneck(r):
    d = {"SA (MXU)": r["sa_avg"], "VPU (softmax)": r["vpu"], "DRAM": r["dram_util"]}
    return max(d, key=d.get)
bott = {}
for r in rows:
    bott[bottleneck(r)] = bott.get(bottleneck(r), 0) + 1

# H2a per-head cycles MHA vs GQA (matched B,S,D)
def find(mode,gqa,B,Hq,Hkv,S,D):
    for r in rows:
        if r["mode"]==mode and r["gqa"]==str(gqa) and r["B"]==B and r["Hq"]==Hq and r["Hkv"]==Hkv and r["S"]==S and r["D"]==D:
            return r
    return None
perhead = []
for (B,S,D) in [(1,256,128),(1,512,128),(4,512,128)]:
    entry = {"cfg": f"B{B} S{S} D{D}"}
    for lbl,(gqa,Hq,Hkv) in {"MHA H8":(0,8,8),"GQA Hq8":(1,8,1),"GQA Hq16":(1,16,1),"GQA Hq64":(1,64,8)}.items():
        r = find("prefill",gqa,B,Hq,Hkv,S,D)
        entry[lbl] = round(r["c"]/r["Hq"],0) if r else None
    perhead.append(entry)

# H2b: KV DMA reads & cycles, MHA (ratio1) vs GQA Hq8/Hkv1 (ratio8) at matched Hq=8
def ratio_pts(mode):
    out=[]
    for (B,S,D) in [(1,256,128),(1,512,128)]:
        pts=[]
        for lbl,(gqa,Hq,Hkv,ratio) in {"MHA (ratio1)":(0,8,8,1),"GQA (ratio8)":(1,8,1,8)}.items():
            r = find(mode,gqa,B,Hq,Hkv,S,D)
            if r: pts.append({"lbl":lbl,"ratio":ratio,"reads":r["dram_responses"],"cycles":r["c"],
                              "percyc":round(r["c"]/r["Hq"],0),"dram_util":round(r["dram_util"],1)})
        if len(pts)==2: out.append({"cfg":f"B{B} S{S} D{D}","pts":pts})
    return out
decode_ratio = ratio_pts("decode")
prefill_ratio = ratio_pts("prefill")

# Scaling vs S (MHA only), with utilization -- prefill capped, decode full long-context range
def scaling_series(mode):
    Ss = sorted({r["S"] for r in rows if r["mode"]==mode and r["gqa"]=="0"})
    out=[]
    for S in Ss:
        g=[r for r in rows if r["mode"]==mode and r["gqa"]=="0" and r["S"]==S]
        if not g: continue
        rd=[r["dram_responses"] for r in g if r["dram_responses"] is not None]
        out.append({"S":S, "cyc":int(st.median(r["c"] for r in g)),
                    "reads":int(st.median(rd)) if rd else None,
                    "mxu":round(st.mean(r["sa_avg"] for r in g),1),
                    "vpu":round(st.mean(r["vpu"] for r in g),1),
                    "dram":round(st.mean(r["dram_util"] for r in g),1)})
    return out
scaling = {"prefill": scaling_series("prefill"), "decode": scaling_series("decode")}

data = dict(groups=groups, bottleneck=bott, perhead=perhead, decode_ratio=decode_ratio,
            prefill_ratio=prefill_ratio, scaling=scaling, n=len(rows), peak_bw=PEAK_BW)
json.dump(data, open(JOUT,"w"), indent=2)
print("source CSV  :", SRC)
print("report JSON :", JOUT)
print(json.dumps(data, indent=2))
