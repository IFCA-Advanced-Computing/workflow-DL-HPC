#!/bin/bash
#
#SBATCH --job-name=setup
#SBATCH --partition=wngpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --output=setup.out
#SBATCH --error=setup.err

export PATH=$PATH:/bin/
source /etc/profile.d/modules.sh

module purge
module load OPENMPI/4.0.1
module load CMAKE/3.9.0

export HOROVOD_CMAKE="/gpfs/res_projects/apps/CMAKE/SRC/cmake-3.9.0-Linux-x86_64/bin/cmake"

###export CUDA_HOME="/usr/local/cuda-10.1"
###export HOROVOD_CUDA_HOME=$CUDA_HOME
###export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

export HOROVOD_NCCL_HOME="$HOME/nccl_2.8.4-1+cuda10.2_x86_64/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOROVOD_NCCL_HOME"
export HOROVOD_GPU_ALLREDUCE=NCCL

export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_GLOO=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda create --name multiGPU tensorflow-gpu
conda clean --all

conda activate multiGPU
/gpfs/users/gonzabad/miniconda3/envs/multiGPU/bin/pip install horovod[tensorflow]
horovodrun --check-build
