# Description

Implements the VerifyObject action server
Uses a differentiable renderer (Pytorch3d) to perform an optimization of the scene objects' poses based on their object detection masks (image-space criterion) but also 3D criterion (like plane support/collision)

# Installation
- Run `docker compose build verify_object` to build the docker image
- You should be able to run the service using `docker compose run --rm verify_object`