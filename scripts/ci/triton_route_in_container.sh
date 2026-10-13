#!/usr/bin/env bash
# What the Triton route runs inside the base image: copies the checkout mounted read-only at /src
# to $TORCHSIM_DIR, builds the device extension as docker/Dockerfile does, then runs the sweep with
# the arguments given. Only the build can fail it; the sweep reports. $SWEEP_OUT/device-built says the build held.
set -euo pipefail
SRC="${SRC:-/src}"
DEST="${TORCHSIM_DIR:-/workspace/PyTorchSim}"
OUT="${SWEEP_OUT:-/sweep}"

mkdir -p "$DEST"
tar -C "$SRC" --exclude=./.git --exclude=./.github --exclude=./.claude -cf - . | tar -C "$DEST" -xf -
(cd "$DEST/device" && python -m pip install --no-build-isolation -e .)
touch "$OUT/device-built"

cd "$DEST"
rc=0
python3 scripts/ci/triton_route_sweep.py "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "::warning::the sweep exited $rc: an allowlisted test regressed, or the sweep itself broke (see its report)"
fi
