#!/usr/bin/env bash
# The tag of CI's base image: a hash of everything that decides it -- the compiler pin (the image
# it is built FROM), Dockerfile.base and thirdparty/github-releases.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
{ cat thirdparty/pytorchsim-triton-backend.json; cat thirdparty/github-releases.json; cat Dockerfile.base; } | sha256sum | awk '{print substr($1,1,12)}'
