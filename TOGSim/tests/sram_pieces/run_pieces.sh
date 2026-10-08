#!/bin/bash
# A spad buffer is as large as the producer states it (togsim_spad_buffer_bytes), however many DMAs
# fill or drain it: pieces 32 KB, scatter 16640 B (not its index buffer twice), a tile stored twice
# 32 KB, a state stored every trip 16 KB, a computed state stored every trip 32 KB (48 KB spad);
# a DMA'd buffer the producer does not size is refused.
#     run_pieces.sh [Simulator] [togsim_runtime.h dir]    (defaults: this tree's build and include)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${1:-$HERE/../../build/bin/Simulator}"
INC="${2:-$HERE/../../include}"
SRC_CFG="$HERE/../../../configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
OUT="$(mktemp -d)"
sed -e 's/^core_spad_size_kb:.*/core_spad_size_kb: 48/' \
    -e "s#^ramulator_config_path: \.\./#ramulator_config_path: $HERE/../../../#" "$SRC_CFG" > "$OUT/cfg.yml"
echo "200 0" > "$OUT/cyc.tsv"
printf "1048576\n%.0s" 1 2 3 > "$OUT/trace_tensors.txt"   # tensors 1 MiB apart, as before packed layouts
fail=0
for p in pieces:32768:1 scatter:16640:2 twostore:32768:1 runstate:16384:2 computestate:32768:1; do
  IFS=: read -r name foot disp <<< "$p"
  g++ -std=c++17 -shared -fPIC -O1 -I"$INC" "$HERE/${name}_producer.cpp" -o "$OUT/$name.so" || exit 2
  timeout 120 "$SIM" --config "$OUT/cfg.yml" --trace_so "$OUT/$name.so" --cycle_table "$OUT/cyc.tsv" --log_level trace > "$OUT/$name.log" 2>&1
  rc=$?
  python3 - "$OUT/$name.log" "$rc" "$name" "$foot" "$disp" <<'PY' || fail=1
import re, sys
log, rc, name, foot, disp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = [tuple(map(int, m.groups())) for m in
        re.finditer(r"TILE_SCHEDULED\s*\] spad_footprint=(\d+) max_dispatch=(\d+)", open(log).read())]
print(f"{name}: rc={rc}, scheduled (footprint, max_dispatch) = {rows}")
if rc != "0": sys.exit(f"{name} FAIL: simulator exited non-zero")
if rows != [(foot, disp)] * 2: sys.exit(f"{name} FAIL: expected two work-items of {foot} bytes, max_dispatch {disp}")
print(f"{name} PASS")
PY
done
for c in 0:"spad buffer 1 is DMA'd but the producer states no size" 1:"states no spad buffer sizes"; do
  IFS=: read -r n want <<< "$c"
  g++ -std=c++17 -shared -fPIC -O1 -DUNSIZED_CASE=$n -I"$INC" "$HERE/unsized_producer.cpp" -o "$OUT/unsized_$n.so" || exit 2
  timeout 120 "$SIM" --config "$OUT/cfg.yml" --trace_so "$OUT/unsized_$n.so" --cycle_table "$OUT/cyc.tsv" > "$OUT/unsized_$n.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] && grep -q "$want" "$OUT/unsized_$n.log"; then echo "unsized case $n refused (rc=$rc) PASS"
  else echo "unsized case $n FAIL: rc=$rc, wanted \"$want\""; fail=1; fi
done
echo "logs: $OUT"
exit $fail
