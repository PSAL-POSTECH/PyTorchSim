#!/bin/bash
# Measure TileGraph CONSTRUCTION only (TOGSIM_BUILD_ONLY makes Simulator exit right after).
# Usage: SIM=/path/to/bin/Simulator bash run.sh CI36 MM36
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${SIM:?set SIM=/path/to/TOGSim/build/bin/Simulator}"
CFG="$HERE/config_8x8_nofunc.yml"
export TOGSIM_BUILD_ONLY=1
for c in "$@"; do
  echo "########## $c ##########"
  /usr/bin/time -v "$SIM" --config "$CFG" \
      --trace_so "$HERE/traces/$c/trace.so" \
      --cycle_table "$HERE/traces/$c/trace_cycles.tsv" \
      >/dev/null 2> "$HERE/$c.log"
  echo "  exit=$?"
  echo "  --- last progress lines ---"; grep '\[PROG\]' "$HERE/$c.log" | tail -5
  echo "  --- final totals ---";        grep '\[WI\]'   "$HERE/$c.log" || echo "  [WI] NEVER REACHED: construction did not finish"
  grep 'Maximum resident set size' "$HERE/$c.log" || true
  grep 'Elapsed (wall clock)'      "$HERE/$c.log" || true
done
