# The hub: JupyterHub that starts one container of the tutorial image for each attendee who signs up.
FROM jupyterhub/jupyterhub:5.3.0
RUN pip install --no-cache-dir dockerspawner==14.0.0 jupyterhub-nativeauthenticator==1.3.0
WORKDIR /srv/jupyterhub
