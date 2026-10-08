#!/bin/bash
# One compute at a time per core, and systolic_array_split, on a fixed typed gemm 512 b512 trace (gem5 cycles,
# overlap 0, Issue rows) and a cross-lane trace; every case below names what it pins, "PASS" iff all hold.
#     run_core_serial.sh [Simulator] [togsim_runtime.h dir]    (defaults: this tree's build and include)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${1:-$HERE/../../build/bin/Simulator}"
INC="${2:-$HERE/../../include}"
SRC_CFG="$HERE/../../../configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
OUT="$(mktemp -d)"
sed -e "s#^ramulator_config_path: \.\./#ramulator_config_path: $HERE/../../../#" "$SRC_CFG" > "$OUT/cfg.yml"
{ cat "$OUT/cfg.yml"; echo "systolic_array_split: round_robin"; } > "$OUT/split.yml"
sed -e 's/^num_systolic_array_per_core: .*/num_systolic_array_per_core: 4/' "$OUT/split.yml" > "$OUT/split4.yml"
{ cat "$OUT/cfg.yml"; echo "systolic_array_unit: Msa"; } > "$OUT/msa_nosplit.yml"
sed -e 's/^num_cores: 1$/num_cores: 2/' "$OUT/cfg.yml" > "$OUT/c2.yml"
printf "150000\t0\n100\t0\n" > "$OUT/xlu_cycles.tsv"
printf "300\t0\n100\t0\n" > "$OUT/xlu_c2_cycles.tsv"
for g in g1 g2 x; do mkdir -p "$OUT/$g"; cp "$HERE"/trace_tensors.txt "$HERE"/trace_tensor_shapes.txt "$OUT/$g/"; done
cp "$HERE/trace_shape.txt" "$OUT/g1/trace_shape.txt"; cp "$HERE/trace_shape_2x1.txt" "$OUT/g2/trace_shape.txt"
cp "$HERE/trace_shape.txt" "$OUT/x/trace_shape.txt"
for g in g1 g2; do g++ -std=c++17 -shared -fPIC -O1 -I"$INC" "$HERE/gemm512_typed_producer.cpp" -o "$OUT/$g/trace.so" || exit 2; done
g++ -std=c++17 -shared -fPIC -O1 -I"$INC" "$HERE/xlu_producer.cpp" -o "$OUT/x/trace.so" || exit 2

run() {   # run <name> <trace dir> <cfg> <cycle table> [<unit table>]
  local ut=(); [ $# -ge 5 ] && ut=(--unit_table "$HERE/$5")
  timeout 600 "$SIM" --config "$OUT/$3" --trace_so "$OUT/$2/trace.so" --cycle_table "$4" "${ut[@]}" \
    --log_level trace > "$OUT/$1.log" 2>&1
  echo $? > "$OUT/$1.rc"
}
run serial g1 cfg.yml "$HERE/cycles.tsv" units.tsv
run two_dispatch g2 cfg.yml "$HERE/cycles.tsv" units.tsv
run xlu x cfg.yml "$OUT/xlu_cycles.tsv"
run xlu_c2 x c2.yml "$OUT/xlu_c2_cycles.tsv"
run split g1 split.yml "$HERE/cycles.tsv" units.tsv
run split_one g1 split.yml "$HERE/cycles.tsv" units_one_subtile.tsv
run split_vpu g1 split.yml "$HERE/cycles.tsv" units_vpu700.tsv
run split_short g1 split.yml "$HERE/cycles_cc1100.tsv" units.tsv
run split_no_array g1 split.yml "$HERE/cycles.tsv" units_no_array.tsv
run split4_one_over g1 split4.yml "$HERE/cycles_cc230.tsv" units_one_subtile.tsv
run msa_nosplit g1 msa_nosplit.yml "$HERE/cycles.tsv" units.tsv

python3 - "$OUT" <<'PY'
import re, sys
out = sys.argv[1]
LINE = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[(INST_ISSUED|INST_FINISHED)\s*\]\[INST_ID=(\d+)\] COMP "
                  r"\(compute_type=(\d+) compute_cycle=(\d+)")

def log(name): return open(f"{out}/{name}.log").read()
def rc(name): return int(open(f"{out}/{name}.rc").read())

def comps(name):
    """(core, start, finish, compute_type, cycles) of every compute that finished."""
    issued, finished = {}, {}
    for m in LINE.finditer(log(name)):
        key, rec = (int(m[2]), int(m[4])), (int(m[1]), int(m[5]), int(m[6]))
        (issued if m[3] == "INST_ISSUED" else finished)[key] = rec
    return [(k[0], finished[k][0] - issued[k][2], finished[k][0], issued[k][1], issued[k][2])
            for k in issued if k in finished]

def overlaps(cs):
    """Pairs of same-core computes with cycles whose windows overlap."""
    w = sorted(c for c in cs if c[4] > 0)
    return [(a, b) for a, b in zip(w, w[1:]) if a[0] == b[0] and b[1] < a[2]]

bad = []
def check(label, ok, said):
    print(f"{'ok  ' if ok else 'FAIL'}  {label}: {said}")
    if not ok: bad.append(label)

total = re.search(r"Total execution cycles: (\d+)", log("serial"))
check("K1 --unit_table passes", rc("serial") == 0 and total, f"exit {rc('serial')}, {total[1] if total else 'no'} cycles")
for name in ("serial", "two_dispatch", "xlu", "xlu_c2"):
    cs, ov = comps(name), overlaps(comps(name))
    check(f"K2 {name}: no two computes of a core overlap", rc(name) == 0 and cs and not ov,
          f"exit {rc(name)}, {len(cs)} computes, {len(ov)} overlapping{' e.g. ' + str(ov[0]) if ov else ''}")
cs = [c for c in comps("xlu_c2") if c[4] > 0]
cross = any(a[0] != b[0] and a[1] < b[2] and b[1] < a[2] for a in cs for b in cs)
check("K4 xlu_c2: the two cores still run side by side", {c[0] for c in cs} == {0, 1} and cross,
      f"cores {sorted({c[0] for c in cs})}, cross-core overlap {cross}")
for name, want in (("split", 1077), ("split_one", 1589), ("split_vpu", 1265), ("split_short", 588)):
    mm = sorted({c[4] for c in comps(name) if c[3] == 1})
    check(f"K3 {name}: matmul runs {want}", rc(name) == 0 and mm == [want], f"exit {rc(name)}, matmul {mm}")
for name, said in (("split_no_array", "has no array unit"), ("msa_nosplit", "is ignored without"),
                   ("split4_one_over", "Systolic spread")):
    check(f"K5 {name} refused", rc(name) != 0 and said in log(name), f"exit {rc(name)}")
print("FAIL " + "; ".join(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
PY
rc=$?
echo "logs: $OUT"
exit $rc
