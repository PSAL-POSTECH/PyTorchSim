#!/usr/bin/env bash
# Deterministic short pin for tagging torchsim_base images (thirdparty + base Dockerfile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat thirdparty/github-releases.json; cat Dockerfile.base; } | sha256sum | awk '{print substr($1,1,12)}'
