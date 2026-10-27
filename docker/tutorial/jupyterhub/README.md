# The tutorial's JupyterHub

One host serves every attendee: each signs up in the browser and gets their own container of
`ghcr.io/psal-postech/tutorial-micro2026:latest`, with JupyterLab open in `/workspace`. The slides'
commands go in its terminal (File > New > Terminal); the hands-on's files open in its editor.

## Bring it up

On a Linux x86_64 host with Docker Engine and the Compose v2 plugin:

```bash
git clone -b pytorchsim-triton-backend https://github.com/PSAL-POSTECH/PyTorchSim.git
bash PyTorchSim/docker/tutorial/jupyterhub/setting.sh
```

`setting.sh` creates the `jupyterhub-network` network, pulls the tutorial image and starts the hub
(`docker compose up -d --build`). Attendees open `http://<host>:8888`, click Sign up, choose a name and
a password, then log in.

| Setting | Default | Where |
|---|---|---|
| Hub port | 8888 | `HUB_PORT` when running `setting.sh` |
| Image | `tutorial-micro2026:latest` | `TUTORIAL_IMAGE` |
| Per attendee | 8 CPUs, 32 GB | `jupyterhub_config.py` |

An attendee's container is kept when it stops, so their edits and `build/` survive a log-out.
`docker compose down` in this directory stops the hub; `docker rm -f $(docker ps -aq -f name=jupyter-)`
removes the attendees' containers.
