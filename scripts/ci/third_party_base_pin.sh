#!/usr/bin/env bash
# The tag of CI's base image: a hash of everything that decides it -- the compiler pin (the image
# it is built FROM) and docker/Dockerfile.base.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat third_party/pytorchsim-triton-backend.json; cat docker/Dockerfile.base; } | sha256sum | awk '{print substr($1,1,12)}'
