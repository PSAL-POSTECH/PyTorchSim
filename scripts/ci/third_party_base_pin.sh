#!/usr/bin/env bash
# The tag of CI's base image: a hash of everything that decides it -- the compiler pin (the image
# it is built FROM), docker/Dockerfile.base and third_party/github-releases.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat third_party/pytorchsim-triton-backend.json; cat third_party/github-releases.json; cat docker/Dockerfile.base; } | sha256sum | awk '{print substr($1,1,12)}'
