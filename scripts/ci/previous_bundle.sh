#!/usr/bin/env bash
# The newest pts-bundle on BRANCH with a selection record, whole: record/ to PICKS, out/ and keys/ to PREV (for
# bundle.py --reuse), so the picks and the kernels are one run's. Never fails; with none both stay empty.
#   GH_TOKEN=... REPO=owner/name scripts/ci/previous_bundle.sh PICKS PREV
set -uo pipefail
: "${GH_TOKEN:?a token that reads the actions of this repository}"
: "${REPO:?owner/name}"
: "${BRANCH:=pytorchsim-triton-backend}"
: "${WORKFLOW:=pytorchsim_triton_backend.yml}"
: "${EVENT:=push}"
PICKS=${1:?the directory to write the record to}
PREV=${2:?the directory to write the previous bundle to}
mkdir -p "$PICKS" "$PREV"

api() { curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$@"; }

RUNS=$(api "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW/runs?branch=$BRANCH&event=$EVENT&per_page=30") \
  || { echo "::warning::could not list the bundle runs on $BRANCH: every workload is built"; exit 0; }
for run in $(jq -r --arg me "${GITHUB_RUN_ID:-0}" '.workflow_runs[] | select((.id|tostring) != $me) | .id' <<<"$RUNS"); do
  id=$(api "https://api.github.com/repos/$REPO/actions/runs/$run/artifacts?per_page=100" \
       | jq -r '[.artifacts[] | select(.name == "pts-bundle" and (.expired|not))][0].id // empty') || continue
  [ -n "$id" ] || continue
  zip=$(mktemp) && api -o "$zip" "https://api.github.com/repos/$REPO/actions/artifacts/$id/zip" || { rm -f "$zip"; continue; }
  python3 -I - "$zip" "$PICKS" "$PREV" <<'PY' || { rm -f "$zip"; rm -rf "${PICKS:?}"/* "${PREV:?}"/*; continue; }
import os, sys, zipfile
z, picks, prev = zipfile.ZipFile(sys.argv[1]), sys.argv[2], sys.argv[3]
names = z.namelist()
record = [n for n in names if n.startswith("record/") and n.endswith(".json")]
if not record:
    sys.exit(1)
for n in record:
    with open(os.path.join(picks, os.path.basename(n)), "wb") as f:
        f.write(z.read(n))
for n in names:
    if n.startswith(("out/", "keys/")) and not n.endswith("/"):
        z.extract(n, prev)
PY
  rm -f "$zip"
  echo "bundle of run $run: picks $(ls "$PICKS"), $(ls "$PREV/keys" 2>/dev/null | wc -l) keyed workloads"
  exit 0
done
echo "::warning::no bundle run on $BRANCH left a selection record: every workload is built"
exit 0
