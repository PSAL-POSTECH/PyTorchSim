"""Fold timed picks into this machine's record: python scripts/selection_merge.py SELECTION_JSONL...

Each line a timed run appended ({identity, key, config}) replaces the key's entry. The record keeps
one identity: picks timed with another compiler, machine or TOGSim start a new record.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402,F401  (first: its device autoload installs pytorchsim, which imports template_timing)

from pytorchsim.backend import template_timing  # noqa: E402


def main(paths):
    picks = []
    for path in paths:
        with open(path) as f:
            picks += [json.loads(line) for line in f if line.strip()]
    if not picks:
        sys.exit("no picks in " + " ".join(paths))
    identities = {json.dumps(p["identity"], sort_keys=True) for p in picks}
    if len(identities) != 1:
        sys.exit(f"the picks carry {len(identities)} identities; merge one run's at a time")
    identity = picks[0]["identity"]
    path = template_timing.record_path()
    try:
        with open(path) as f:
            rec = json.load(f)
    except FileNotFoundError:
        rec = None
    if rec is None or rec["identity"] != identity:
        rec = {"identity": identity, "table": {}}
    for p in picks:
        rec["table"][p["key"]] = [p["config"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{path}: {len(rec['table'])} picks")


if __name__ == "__main__":
    main(sys.argv[1:])
