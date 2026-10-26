#!/usr/bin/env bash
# Coprocessor's tutorial branch at third_party/tutorial.json's commit in /workspace/Coprocessor, checked out and built,
# so an attendee's first scripts/build.sh only rebuilds what they changed.
set -euo pipefail
read -r REPO START START_SHA < <(python3 -c "
import json
d = json.load(open('/workspace/PyTorchSim/third_party/tutorial.json'))['coprocessor']
print(d['repository'], d['start']['branch'], d['start']['commit'])")
git clone -q "$REPO" /workspace/Coprocessor
cd /workspace/Coprocessor
git fetch -q origin "$START_SHA"
git checkout -q -B "$START" "$START_SHA"
scripts/build.sh -j "$(nproc)" build
