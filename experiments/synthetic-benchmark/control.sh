#!/bin/bash

# udocker config (if running through udocker)
export UDOCKER="$HOME/udocker" 
export CONTAINER="multigpu-base"

# conda env name (if running in a native environment)
export ENV_NAME="multiGPU"

# Job config
export JOB_NAME="synBen"
export JOB2RUN="./job_udocker.sh" # Select job script based on type of execution (conda vs native)
export DIR_TO_MOUNT="$HOME/experiments/synthetic-benchmark"

# GPUs config
export NUM_GPUS=('1' '2' '3' '4' '5' '6')

# Launch jobs per number of GPUs
for i in "${NUM_GPUS[@]}"
  do
    echo "Num GPUs:" $i
    export GPUS=$i

    if [ $i -eq 1 ]
      then
        export NTASKS_PER_NODE=1
      else
        export NTASKS_PER_NODE=2
      fi
    
    # Specific Slurm directives for CSIC deep learning computer infrastructure
    SLURM_PARAMS="--job-name=${JOB_NAME}_${ec} --partition=wngpu --gres=gpu:2 --ntasks=$i --tasks-per-node=$NTASKS_PER_NODE --time=02:00:00 --mem-per-cpu=64000 --output=${JOB_NAME}_GPU${i}_${ec}.out --error=${JOB_NAME}_GPU${i}_${ec}.err"

    sbatch $SLURM_PARAMS $JOB2RUN --wait
    wait
  done
