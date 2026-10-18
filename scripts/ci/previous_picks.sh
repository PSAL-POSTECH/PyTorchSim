#!/usr/bin/env bash
# The selection record the newest bundle job on BRANCH left (record/ in its pts-bundle artifact), as OUT/<device>.json:
# the bundle takes a recorded config without timing it, and times only what the record lacks or no longer holds
# (another compiler, machine or TOGSim voids the record; a changed template, its entries). Leaves OUT empty when
# there is none, and never fails the job: without it every GEMM is timed.
#   GH_TOKEN=... REPO=owner/name scripts/ci/previous_picks.sh OUT
set -uo pipefail
: "${GH_TOKEN:?a token that reads the actions of this repository}"
: "${REPO:?owner/name}"
: "${BRANCH:=pytorchsim-triton-backend}"
: "${WORKFLOW:=pytorchsim_triton_backend.yml}"
: "${EVENT:=push}"
OUT=${1:?the directory to write}
mkdir -p "$OUT"

api() { curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$@"; }

RUNS=$(api "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW/runs?branch=$BRANCH&event=$EVENT&per_page=30") \
  || { echo "::warning::could not list the bundle runs on $BRANCH: every GEMM is timed"; exit 0; }
for run in $(jq -r --arg me "${GITHUB_RUN_ID:-0}" '.workflow_runs[] | select((.id|tostring) != $me) | .id' <<<"$RUNS"); do
  id=$(api "https://api.github.com/repos/$REPO/actions/runs/$run/artifacts?per_page=100" \
       | jq -r '[.artifacts[] | select(.name == "pts-bundle" and (.expired|not))][0].id // empty') || continue
  [ -n "$id" ] || continue
  api -o "$OUT/a.zip" "https://api.github.com/repos/$REPO/actions/artifacts/$id/zip" || continue
  python3 - "$OUT/a.zip" "$OUT" <<'EOF' || { rm -f "$OUT"/*; continue; }
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.startswith("record/") and n.endswith(".json")]
if not names:
    sys.exit(1)
for n in names:
    with open(os.path.join(sys.argv[2], os.path.basename(n)), "wb") as f:
        f.write(z.read(n))
EOF
  rm -f "$OUT/a.zip"
  echo "picks of run $run: $(ls "$OUT")"
  exit 0
done
echo "::warning::no bundle run on $BRANCH left a selection record: every GEMM is timed"
exit 0
