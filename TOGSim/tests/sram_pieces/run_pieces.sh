#!/bin/bash
# A buffer filled (or drained) by several DMA pieces is sized by their sum, not by the last piece
# (pieces: footprint 32 KB, each work-item alone on a 48 KB spad); an indirect store sizes the
# buffer it drains, not its index buffer (scatter: 16640 B, two work-items share the spad).
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
fail=0
for p in pieces:32768:1 scatter:16640:2; do
  IFS=: read -r name foot disp <<< "$p"
  g++ -std=c++17 -shared -fPIC -O1 -I"$INC" "$HERE/${name}_producer.cpp" -o "$OUT/$name.so" || exit 2
  "$SIM" --config "$OUT/cfg.yml" --trace_so "$OUT/$name.so" --cycle_table "$OUT/cyc.tsv" --log_level trace > "$OUT/$name.log" 2>&1
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
echo "logs: $OUT"
exit $fail
