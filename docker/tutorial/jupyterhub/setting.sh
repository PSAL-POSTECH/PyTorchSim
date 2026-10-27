#!/usr/bin/env bash
# Brings the hub up on port ${HUB_PORT:-8888}: the network its containers share, then the hub itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker network inspect jupyterhub-network >/dev/null 2>&1 || docker network create jupyterhub-network
docker pull "${TUTORIAL_IMAGE:-ghcr.io/psal-postech/tutorial-micro2026:latest}"
docker compose up -d --build
