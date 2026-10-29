#!/usr/bin/env bash
# Coprocessor's tutorial branch at third_party/tutorial.json's commit in /workspace/Coprocessor, checked out and built
# with Hands-on 1's answer, so Hands-on 2 and 3 run before Hands-on 1 is done. The start, whose steps are commented
# out, does not build; it is put back after the build. The branch tracks origin, so a git pull brings a later fix.
# The compiler times a model on the gem5 machine under its own Coprocessor checkout
# (third_party/Coprocessor/examples/<model>/gem5/), which has none for the tutorial's model xlane: a link to the
# attendee's examples/xlane puts it there, and a pull of the tutorial branch updates both.
set -euo pipefail
read -r REPO START START_SHA < <(python3 -c "
import json
d = json.load(open('/workspace/PyTorchSim/third_party/tutorial.json'))['coprocessor']
print(d['repository'], d['start']['branch'], d['start']['commit'])")
git clone -q "$REPO" /workspace/Coprocessor
cd /workspace/Coprocessor
git fetch -q origin "$START_SHA"
git checkout -q -B "$START" "$START_SHA"
git branch -q --set-upstream-to="origin/$START"
cp examples/xlane_reduce/answer/xlane_reduce.hpp examples/xlane_reduce/xlane_reduce.hpp
scripts/build.sh -j "$(nproc)" build
git checkout -q -- examples/xlane_reduce/xlane_reduce.hpp
ln -s /workspace/Coprocessor/examples/xlane /workspace/pytorchsim-triton-compiler/third_party/Coprocessor/examples/xlane
