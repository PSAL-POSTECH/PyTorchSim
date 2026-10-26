#!/usr/bin/env bash
# vcix-accelerator's tutorial branches at third_party/tutorial.json's commits in /workspace/vcix-accelerator, the start
# checked out and built, so an attendee's first scripts/build.sh only rebuilds what they changed.
set -euo pipefail
read -r REPO START START_SHA SOLUTION SOLUTION_SHA < <(python3 -c "
import json
d = json.load(open('/workspace/PyTorchSim/third_party/tutorial.json'))['vcix_accelerator']
print(d['repository'], d['start']['branch'], d['start']['commit'], d['solution']['branch'], d['solution']['commit'])")
git clone -q "$REPO" /workspace/vcix-accelerator
cd /workspace/vcix-accelerator
git fetch -q origin "$START_SHA" "$SOLUTION_SHA"
git branch -f "$SOLUTION" "$SOLUTION_SHA"
git checkout -q -B "$START" "$START_SHA"
scripts/build.sh -j "$(nproc)" build
