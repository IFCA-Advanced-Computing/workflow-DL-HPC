#!/bin/bash

# udocker executable
UDOCKER="$HOME/udocker"

# Pull image fom DockerHub
$UDOCKER pull gonzabad/multigpu-horovod:downscaling

# Create the container
$UDOCKER create --name=multigpu-downscaling gonzabad/multigpu-horovod:downscaling

# Add the GPU libraries to the container
$UDOCKER setup --nvidia --force multigpu-downscaling

# Run it and install OpenMPI and Horovod
$UDOCKER run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-downscaling

