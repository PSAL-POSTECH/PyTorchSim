#!/usr/bin/env bash
# Post-condition for run_speedup.sh: at least one (workload, config) result
# with a numeric average. Run after cycle_validation logs + run_speedup.
set -euo pipefail

if [[ -z "${TORCHSIM_DIR:-}" ]]; then
  echo "check_speedup_smoke: TORCHSIM_DIR is not set" >&2
  exit 1
fi

results="${TORCHSIM_DIR}/experiments/artifact/speedup/results"
if [[ ! -d "$results" ]]; then
  echo "check_speedup_smoke: missing results dir: $results" >&2
  exit 1
fi

n_ok=0
shopt -s nullglob
files=("$results"/*.txt)
if ((${#files[@]} == 0)); then
  echo "check_speedup_smoke: no .txt under $results" >&2
  exit 1
fi
for f in "${files[@]}"; do
  if grep -qE "Average simulation time[[:space:]]*=[[:space:]]*[0-9]+([.][0-9]+)?" "$f"; then
    n_ok=$((n_ok + 1))
  fi
done

if (( n_ok < 1 )); then
  echo "check_speedup_smoke: no .txt in $results with a numeric Average simulation time" >&2
  exit 1
fi
echo "check_speedup_smoke: OK ($n_ok result file(s) with a numeric average)"
