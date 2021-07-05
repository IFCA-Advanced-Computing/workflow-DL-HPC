#!/bin/bash

# udocker executable
UDOCKER="$HOME/udocker"

# Pull image fom DockerHub
$UDOCKER pull gonzabad/multigpu-horovod:base

# Create the container
$UDOCKER create --name=multigpu-base gonzabad/multigpu-horovod:base

# Add the GPU libraries to the container
$UDOCKER setup --nvidia --force multigpu-base

# Run it and install OpenMPI and Horovod
$UDOCKER run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-base

