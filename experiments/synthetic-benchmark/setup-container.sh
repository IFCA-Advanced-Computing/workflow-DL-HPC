#!/bin/bash

# Pull image fom DockerHub
udocker pull gonzabad/multigpu-horovod:base

# Create the container
udocker create --name=multigpu-base gonzabad/multigpu-horovod:base

# Add the GPU libraries to the container
udocker setup --nvidia --force multigpu-base

# Run it and install OpenMPI and Horovod
udocker run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-base

