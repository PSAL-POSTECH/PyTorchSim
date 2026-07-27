#!/usr/bin/env bash
# Deterministic short pin for tagging torchsim_tnpu_base images.
# Mirrors thirdparty_base_pin.sh, over the tnpu manifest + its Dockerfile, so the
# ~1.8 GiB toolchain layer is rebuilt only when one of those two actually moves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat thirdparty/triton-npu.json; cat Dockerfile.tnpu; } | sha256sum | awk '{print substr($1,1,12)}'
