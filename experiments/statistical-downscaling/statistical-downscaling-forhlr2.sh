#!/bin/bash

# Project directory
export SD_DIRECTORY="$HOME/experiments/statistical-downscaling"

# udocker container name
export CONTAINER="multigpu-downscaling"

# Job config
export JOB_NAME="multiGPU-SD"
export JOB2RUN="$SD_DIRECTORY/SD-job.sh"

# GPU config
export NUM_NODES=1
export NTASKS=('1' '2' '3' '4')

for i in "${NTASKS[@]}"
  do
    echo '****************************************************************************************************'
    echo 'Training model on' $i 'GPUs'
    
    # Specific Slurm directives for ForHLR2 supercomputer
    SLURM_PARAMS="-p visu --job-name=${JOB_NAME}_GPU${i} --nodes=${NUM_NODES} --ntasks-per-node=${i} --time=24:00:00 --output=${JOB_NAME}_GPU${i}.out --error=${JOB_NAME}_GPU${i}.err"

    # Job launching
    module load mpi/openmpi/4.0
    sbatch $SLURM_PARAMS $JOB2RUN
  done
