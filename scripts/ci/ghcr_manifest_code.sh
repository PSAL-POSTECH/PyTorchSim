#!/usr/bin/env bash
# HTTP status of ghcr.io's manifest for IMAGE (repo:tag), with $TOKEN; 200 means the tag exists.
set -euo pipefail
ref="${1#ghcr.io/}"; repo="${ref%:*}"; tag="${ref##*:}"
curl -s -o /dev/null -w '%{http_code}' -I \
  -H "Authorization: Bearer $(printf %s "$TOKEN" | base64 -w0)" \
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://ghcr.io/v2/$repo/manifests/$tag"
