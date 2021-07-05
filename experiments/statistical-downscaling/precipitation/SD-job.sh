#!/bin/bash

MPI_PARAMS="-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO \
-x HOROVOD_MPI_THREADS_DISABLE=1 \
-mca pmix_server_usock_connections 1 \
-mca pml ob1 -mca btl ^openib"

$UDOCKER setup --nvidia --force $CONTAINER

nvidia-modprobe -u -c=0
nvidia-smi

mpirun $MPI_PARAMS \
$UDOCKER run --hostenv --hostauth --user=$USER \
         --volume=$SD_DIRECTORY:/examples/ $CONTAINER \
	 python train_model.py
