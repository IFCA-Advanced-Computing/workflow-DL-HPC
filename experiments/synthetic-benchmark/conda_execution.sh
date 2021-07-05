#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
python $DIR_TO_MOUNT/tensorflow2_synthetic_benchmark.py \
        --model $MODEL --batch-size $BATCH
