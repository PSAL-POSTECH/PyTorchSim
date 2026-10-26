# Each attendee signs up at the hub, and gets a container of the tutorial image with JupyterLab in /workspace:
# a terminal for the slides' commands and an editor for the hands-on's files. The container is kept when it stops.
import os

c = get_config()  # noqa: F821

c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"
c.DockerSpawner.image = os.environ.get("TUTORIAL_IMAGE", "ghcr.io/psal-postech/tutorial-micro2026:latest")
c.DockerSpawner.network_name = os.environ.get("DOCKER_NETWORK_NAME", "jupyterhub-network")
c.DockerSpawner.remove = False
c.DockerSpawner.cmd = ["jupyterhub-singleuser", "--allow-root"]
c.DockerSpawner.notebook_dir = "/workspace"
c.DockerSpawner.environment = {"SHELL": "/bin/bash"}
c.DockerSpawner.extra_host_config = {"ipc_mode": "host"}
c.DockerSpawner.mem_limit = os.environ.get("TUTORIAL_MEM", "32G")
c.DockerSpawner.cpu_limit = float(os.environ.get("TUTORIAL_CPUS", "8"))
c.Spawner.default_url = "/lab"
c.Spawner.ip = "0.0.0.0"
c.Spawner.start_timeout = 600
c.Spawner.http_timeout = 300

c.JupyterHub.hub_ip = "jupyterhub"
c.JupyterHub.hub_port = 8081

c.JupyterHub.authenticator_class = "nativeauthenticator.NativeAuthenticator"
c.Authenticator.admin_users = {"admin"}
c.NativeAuthenticator.open_signup = True
c.Authenticator.allow_all = True
