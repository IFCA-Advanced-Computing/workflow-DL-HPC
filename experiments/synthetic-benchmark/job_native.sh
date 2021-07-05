#!/bin/bash

source /etc/profile.d/modules.sh
module purge
module load OPENMPI/4.0.1

MODELS_TO_RUN=('InceptionV3' 'ResNet50' 'ResNet101')
BATCH_SIZES=(256 256 128)

FILE_TRAINING="$DIR_TO_MOUNT/conda_execution.sh"

echo "Using" $ENV_NAME "conda environment"

MPI_PARAMS="-np $GPUS -npernode $NTASKS_PER_NODE \
-bind-to none -map-by slot \
-x NCCL_DEBUG=VERSION \
-mca pml ob1 -mca btl ^openib"

for j in "${!MODELS_TO_RUN[@]}"
  do
    echo '****************************************************************************************************'
    echo "Training model" ${MODELS_TO_RUN[$j]} "with bs" ${BATCH_SIZES[$j]}
    export MODEL=${MODELS_TO_RUN[$j]}
    export BATCH=${BATCH_SIZES[$j]}
    mpirun $MPI_PARAMS FILE_TRAINING
    wait
  done
  
