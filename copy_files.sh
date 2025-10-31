#!/usr/bin/env bash


DOCKER_ID=$(docker ps -q --filter "name=miro-docker")
docker cp cropped_pictures "$DOCKER_ID:/root/mdk/bin/shared"
docker cp look_miro.py "$DOCKER_ID:/root/mdk/bin/shared"
docker cp take_images.world "$DOCKER_ID:/root/mdk/sim/worlds"
docker cp take_images.py "$DOCKER_ID:/root/mdk/bin/shared"