#!/bin/bash

# Pull image fom DockerHub
udocker pull gonzabad/multigpu-horovod:downscaling

# Create the container
udocker create --name=multigpu-downscaling gonzabad/multigpu-horovod:downscaling

# Add the GPU libraries to the container
udocker setup --nvidia --force multigpu-downscaling

# Run it and install OpenMPI and Horovod
udocker run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-downscaling

