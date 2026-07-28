if [ -z "$(docker network ls | grep jupyterhub-network)" ]; then
    docker network create jupyterhub-network
fi

docker compose up -d --build