#!/usr/bin/env bash
# Emit TNPU_LLVM23_ASSET_ID / TNPU_SPIKE_ASSET_ID / TNPU_RUNTIME_ASSET_ID lines
# for appending to GITHUB_ENV.
#
# Same shape as thirdparty_github_asset_env.sh, with one difference that matters:
# PSAL-POSTECH/triton-npu is PRIVATE, so the default `secrets.GITHUB_TOKEN` (which
# is scoped to the PyTorchSim repo) cannot read its releases. Pass a PAT with read
# access instead -- the workflow puts secrets.TNPU_TOKEN here.
#
# Release assets of a private repo are also only reachable through the API asset
# id, not the /releases/download/ URL, which is why ids are resolved at all
# (thirdparty/github-releases.json documents the same constraint).
#
# Requires: jq, curl, GITHUB_TOKEN, repo root as cwd or GITHUB_WORKSPACE.
set -euo pipefail
ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
MANIFEST="${ROOT}/thirdparty/triton-npu.json"
if [ ! -f "$MANIFEST" ]; then
  echo "Missing tnpu manifest: $MANIFEST" >&2
  exit 1
fi
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is not set (needs a PAT that can read the private" >&2
  echo "PSAL-POSTECH/triton-npu; the default Actions token cannot)" >&2
  exit 1
fi

REPO=$(jq -r '.triton_npu.repository' "$MANIFEST")
TAG=$(jq -r '.triton_npu.release_tag' "$MANIFEST")
OWNER="${REPO%%/*}"
NAME="${REPO##*/}"

if [ "$TAG" = "latest" ]; then
  API_URL="https://api.github.com/repos/${OWNER}/${NAME}/releases/latest"
else
  API_URL="https://api.github.com/repos/${OWNER}/${NAME}/releases/tags/${TAG}"
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
if ! curl -fsS -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "$API_URL" -o "$TMP"; then
  echo "Failed to read release ${TAG} of ${REPO}." >&2
  echo "Either the token cannot see the repo, or the release does not exist" >&2
  echo "yet -- as of writing, ${REPO} carries no releases and the toolchain" >&2
  echo "assets live only on the upstream fork. See thirdparty/triton-npu.json." >&2
  exit 1
fi

emit() {  # emit <asset-name> <var>
  local id
  id=$(jq -r --arg n "$1" '.assets[] | select(.name == $n) | .id' "$TMP" | head -n1)
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    echo "Release ${TAG} of ${REPO} has no asset named '$1'" >&2
    exit 1
  fi
  echo "$2=${id}"
}

emit llvm23-install.tar.gz  TNPU_LLVM23_ASSET_ID
emit spike-install.tar.gz   TNPU_SPIKE_ASSET_ID
emit triton-runtime.tar.gz  TNPU_RUNTIME_ASSET_ID
