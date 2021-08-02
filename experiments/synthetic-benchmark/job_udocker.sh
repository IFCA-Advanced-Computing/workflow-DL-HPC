#!/bin/bash

# Load modules (specific of the CSIC infrastructure)
source /etc/profile.d/modules.sh
module purge
module load OPENMPI/4.1.0

# Models to run and its respective batch sizes
MODELS_TO_RUN=('InceptionV3' 'ResNet50' 'ResNet101')
BATCH_SIZES=(256 256 128)

# MPI params
MPI_PARAMS="-np $GPUS \
-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO \
-x HOROVOD_MPI_THREADS_DISABLE=1 \
-mca pml ob1 -mca btl ^openib"

# Add GPUs libraries to the container
udocker setup --nvidia --force $CONTAINER
nvidia-modprobe -u -c=0

# Run the different benchmarks as jobs
for j in "${!MODELS_TO_RUN[@]}"
  do
    echo '****************************************************************************************************'
    echo "Training model" ${MODELS_TO_RUN[$j]} "with bs" ${BATCH_SIZES[$j]}
    mpirun $MPI_PARAMS \
    udocker run --hostenv --hostauth --user=$USER \
    -v $DIR_TO_MOUNT/:/home/ $CONTAINER \
    python /home/tensorflow2_synthetic_benchmark.py --model ${MODELS_TO_RUN[$j]} --batch-size ${BATCH_SIZES[$j]}
    wait
  done

