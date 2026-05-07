# PyTorchSim Tutorial: JupyterHub Setup Guide

This directory contains the **JupyterHub** environment used to serve the
[PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim) tutorial to multiple
users concurrently.

- Spawner image: `ghcr.io/psal-postech/torchsim-tutorial:ispass2026`
- Hub port: `8888` (host) → `8000` (container)

---

## 1. Layout

```
tutorial/jupyterhub/
├── Dockerfile             # JupyterHub image (DockerSpawner + NativeAuthenticator)
├── Dockerfile.tutorial    # Single-user tutorial image (built by CI, pushed to GHCR)
├── docker-compose.yml     # Compose file that brings up the Hub
├── jupyterhub_config.py   # Spawner / authenticator / resource-limit settings
└── setting.sh             # Bootstrap script: create network + compose up
```

---

## 2. Dependencies

### Host OS
- Linux x86_64 (Ubuntu 22.04 recommended)
- Enough disk and memory for concurrent users (each user can use up to 32 GB
  RAM and 8 CPUs — see `jupyterhub_config.py`)

### Required software
| Component | Version | Notes |
|---|---|---|
| Docker Engine | 20.10+ | The `docker` CLI must be available |
| Docker Compose Plugin | v2+ | The compose file uses the V2 subcommand (`docker compose`); the legacy `docker-compose` (V1) is not guaranteed to work |

> Compose **V2** is required. On Ubuntu:
>
> ```bash
> sudo apt-get update
> sudo apt-get install -y docker.io docker-compose-plugin
> ```

### Images (pulled automatically)
- **JupyterHub itself** is built locally from `Dockerfile` when you run
  `setting.sh` (tagged `my-jupyterhub-image`).
- **Tutorial single-user image** is `ghcr.io/psal-postech/torchsim-tutorial:ispass2026`.
  Docker pulls it the first time a user is spawned. Pulling it ahead of time
  avoids a long delay on the first login:

  ```bash
  docker pull ghcr.io/psal-postech/torchsim-tutorial:ispass2026
  ```

---

## 3. Quick Start

```bash
# 1) Clone the repo and move into this directory
git clone https://github.com/PSAL-POSTECH/PyTorchSim.git
cd PyTorchSim/tutorial/jupyterhub

# 2) (Optional) Pre-pull the tutorial image
docker pull ghcr.io/psal-postech/torchsim-tutorial:ispass2026

# 3) Bring up JupyterHub
bash setting.sh
```

`setting.sh` performs two steps:

1. Creates the `jupyterhub-network` Docker network if it doesn't exist.
2. Runs `docker compose up -d --build`, which builds the Hub image and starts
   it in the background.

Once the Hub is up, open a browser at:

```
http://<your-host>:8888
```
