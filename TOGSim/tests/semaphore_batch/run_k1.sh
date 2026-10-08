#!/bin/bash
# K1: a barrier on a semaphore two async loads signaled releases only after BOTH arrive; a
# re-wait leaves an in-place compute its buffer's writer; a trace that breaks the semaphore
# contract (K1 cases 1-3, rewait case 1) is refused with a non-zero exit.
#     run_k1.sh [Simulator] [togsim_runtime.h dir]    (defaults: this tree's build and include)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${1:-$HERE/../../build/bin/Simulator}"
INC="${2:-$HERE/../../include}"
CFG="$HERE/../../../configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
OUT="$(mktemp -d)"
echo "200 0" > "$OUT/cyc.tsv"
fail=0

run_case() {
  g++ -std=c++17 -shared -fPIC -O1 -DK1_CASE="$1" -I"$INC" "$HERE/k1_producer.cpp" -o "$OUT/k1_$1.so" || exit 2
  "$SIM" --config "$CFG" --trace_so "$OUT/k1_$1.so" --cycle_table "$OUT/cyc.tsv" --log_level trace > "$OUT/k1_$1.log" 2>&1
}

run_case 0
python3 - "$OUT/k1_0.log" <<'PY' || fail=1
import re, sys
ev = {}
for line in open(sys.argv[1]):
    m = re.search(r"\]\s*\[(\d+)\]\[Core 0\]\[(DRAM_RESP_DONE|INST_FINISHED)\s*\]\[INST_ID=(\d+)\] (MOVIN|MEMORY_BAR|COMP)", line)
    if m: ev[(m.group(2), int(m.group(3)))] = int(m.group(1))
big, small = ev.get(("DRAM_RESP_DONE", 0)), ev.get(("DRAM_RESP_DONE", 1))
bar, comp = ev.get(("INST_FINISHED", 2)), ev.get(("INST_FINISHED", 3))
print(f"K1: big load arrives @{big}, small load arrives @{small}, barrier finishes @{bar}, compute finishes @{comp}")
if None in (big, small, bar, comp): sys.exit("K1 FAIL: missing events in the trace log")
if not small < big: sys.exit("K1 FAIL: scenario lost -- the second load no longer arrives first")
if bar < max(big, small): sys.exit(f"K1 FAIL: barrier released @{bar} before the first load arrived @{big}")
print("K1 PASS")
PY

abi=$(grep -o "define TOGSIM_ABI_VERSION [0-9]*" "$INC/togsim_runtime.h" | grep -o "[0-9]*$")
[ "$abi" -ge 14 ] && cases="1 2 3" || cases=""
for c in $cases; do
  run_case $c; rc=$?
  msg=$(grep -o "\[TOGSim-trace\] [a-z].*" "$OUT/k1_$c.log" | head -1)
  if [ "$rc" -ne 0 ] && [ -n "$msg" ]; then echo "K1 case $c refused (rc=$rc): $msg"
  else echo "K1 case $c FAIL: rc=$rc, no refusal message"; fail=1; fi
done

run_rewait() {
  g++ -std=c++17 -shared -fPIC -O1 -DREWAIT_CASE="$1" -I"$INC" "$HERE/rewait_producer.cpp" -o "$OUT/rewait_$1.so" || exit 2
  "$SIM" --config "$CFG" --trace_so "$OUT/rewait_$1.so" --cycle_table "$OUT/rewait_cyc.tsv" --log_level trace > "$OUT/rewait_$1.log" 2>&1
}

if [ "$abi" -ge 14 ]; then
  printf "2000 0\n100 0\n" > "$OUT/rewait_cyc.tsv"
  run_rewait 0
  python3 - "$OUT/rewait_0.log" <<'PY' || fail=1
import re, sys
ev = {}
for line in open(sys.argv[1]):
    m = re.search(r"\]\s*\[(\d+)\]\[Core 0\]\[(INST_ISSUED|INST_FINISHED)\s*\]\[INST_ID=(\d+)\] COMP", line)
    if m: ev[(m.group(2), int(m.group(3)))] = int(m.group(1))
f_done, g_issue = ev.get(("INST_FINISHED", 2)), ev.get(("INST_ISSUED", 3))
print(f"rewait: in-place compute finishes @{f_done}, the reader after the re-wait issues @{g_issue}")
if None in (f_done, g_issue): sys.exit("rewait FAIL: missing events in the trace log")
if g_issue < f_done: sys.exit(f"rewait FAIL: the reader issued @{g_issue} before the in-place write finished @{f_done}")
print("rewait PASS")
PY
  run_rewait 1; rc=$?
  msg=$(grep -o "\[TOGSim-trace\] [a-z].*" "$OUT/rewait_1.log" | head -1)
  if [ "$rc" -ne 0 ] && [ -n "$msg" ]; then echo "rewait case 1 refused (rc=$rc): $msg"
  else echo "rewait case 1 FAIL: rc=$rc, no refusal message"; fail=1; fi
fi

echo "logs: $OUT"
exit $fail
