#!/usr/bin/env bash
# Emit GEM5_ASSET_ID, LLVM_ASSET_ID, SPIKE_ASSET_ID lines for appending to GITHUB_ENV.
# Requires: jq, curl, GITHUB_TOKEN, repo root as cwd or GITHUB_WORKSPACE.
set -euo pipefail
ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
MANIFEST="${ROOT}/thirdparty/github-releases.json"
if [ ! -f "$MANIFEST" ]; then
  echo "Missing thirdparty manifest: $MANIFEST" >&2
  exit 1
fi
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is not set" >&2
  exit 1
fi

thirdparty_asset_id() {
  local key="$1"
  local out_var="$2"
  local repo release_tag asset_name owner name api_url tmp id
  repo=$(jq -r --arg k "$key" '.[$k].repository' "$MANIFEST")
  release_tag=$(jq -r --arg k "$key" '.[$k].release_tag' "$MANIFEST")
  asset_name=$(jq -r --arg k "$key" '.[$k].asset_name // ""' "$MANIFEST")
  owner="${repo%%/*}"
  name="${repo##*/}"
  if [ "$release_tag" = "latest" ]; then
    api_url="https://api.github.com/repos/${owner}/${name}/releases/latest"
  else
    api_url="https://api.github.com/repos/${owner}/${name}/releases/tags/${release_tag}"
  fi
  tmp=$(mktemp)
  if ! curl -fsS -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$api_url" -o "$tmp"; then
    echo "Failed to fetch release metadata for ${key} (${owner}/${name}, ${release_tag})" >&2
    rm -f "$tmp"
    exit 1
  fi
  if [ -n "$asset_name" ]; then
    id=$(jq -r --arg n "$asset_name" '.assets[] | select(.name == $n) | .id' "$tmp" | head -n1)
  else
    id=$(jq -r '.assets[0].id' "$tmp")
  fi
  rm -f "$tmp"
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    echo "Could not resolve asset id for ${key} (${owner}/${name}, tag=${release_tag}, asset_name=${asset_name:-<first>})" >&2
    exit 1
  fi
  echo "${out_var}=${id}"
}

thirdparty_asset_id gem5 GEM5_ASSET_ID
thirdparty_asset_id llvm_project LLVM_ASSET_ID
thirdparty_asset_id spike SPIKE_ASSET_ID
