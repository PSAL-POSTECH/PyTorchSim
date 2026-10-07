#!/usr/bin/env bash
# Deterministic short pin for tagging torchsim_ptb_base images.
# Mirrors thirdparty_base_pin.sh, over the PTB manifest + its Dockerfile, so the
# ~1.8 GiB toolchain layer is rebuilt only when one of those two actually moves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat thirdparty/pytorchsim-triton-backend.json; cat Dockerfile.ptb; } | sha256sum | awk '{print substr($1,1,12)}'
