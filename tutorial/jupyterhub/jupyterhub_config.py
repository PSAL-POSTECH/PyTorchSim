import os

c = get_config()

# ------------------------------------------------------------------------------
# Spawner config
# ------------------------------------------------------------------------------
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = "ghcr.io/psal-postech/torchsim-tutorial:ispass2026"
c.DockerSpawner.environment = {'SHELL': '/bin/bash'}

# Resource limit
c.DockerSpawner.mem_limit = '32G'
c.DockerSpawner.cpu_limit = 8.0

c.DockerSpawner.network_name = 'jupyterhub-network'
c.Spawner.default_url = '/lab'
c.Spawner.ip = '0.0.0.0'
c.DockerSpawner.remove = False
c.DockerSpawner.cmd = ["jupyterhub-singleuser", "--allow-root"]

c.JupyterHub.authenticator_class = 'nativeauthenticator.NativeAuthenticator'
c.Authenticator.admin_users = {'admin'}

c.JupyterHub.hub_ip = 'jupyterhub'
c.JupyterHub.hub_port = 8081

c.NativeAuthenticator.open_signup = True
c.NativeAuthenticator.allow_all = True
