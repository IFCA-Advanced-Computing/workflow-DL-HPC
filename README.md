# A Container Based Workflow for Distributed Training of Deep Learning Algorithms in HPC Clusters

This repository includes all the code and steps required to reproduce the results of the paper *A Container Based Workflow for Distribute Training of Deep Learning Algorithms in HPC Clusters*

## Prerequisites

To successfully run these experiments, access to an HPC cluster is required. In this cluster users must install udocker ([installation manual](https://indigo-dc.gitbook.io/udocker/installation_manual)). Its dependencies are minimal and common in most Linux distributions, making it easily compatible with most HPC clusters. To install it via GitHub the following needs to be executed:

 ```
# Get the udocker repository and add udocker to PATH
git clone --depth=1 https://github.com/indigo-dc/udocker.git
(cd udocker/udocker; ln -s maincmd.py udocker)
export PATH=`pwd`/udocker/udocker:$PATH

# Install udocker
udocker install
 ```

## Experiments

To prove the feasibility of the proposed workflow we have designed two [experiments](https://github.com/jgonzalezab/workflow-DL-HPC/tree/main/experiments).

### TensorFlow Benchmark

This experiment trains InceptionV3, ResNet50 and ResNet101 models on synthetic data. Its purpose is to verify the correct scalability of the proposed workflow. In order to perform this comparison we will train the models on two different environments: Container Based and Native.

#### Container Based environment

This environment is based on the proposed workflow. We will run the distributed training of the models via udocker containerization. What we need first is an image with all the software needed to run our experiments. A [Dockerfile](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/Dockerfile) with these requirements have been developed using Docker. The corresponding image has been uploaded to [DockerHub](https://hub.docker.com/layers/gonzabad/multigpu-horovod/base/images/sha256-68bfe1ab5d0b36a080e7651066acaafacf5b8901ab5c653eb4b0cc7adec5753f?context=explore) in order to make it accessible through internet. To create a container of this image in the cluster we need to execute the following code:

 ```
# Pull image fom DockerHub
udocker pull gonzabad/multigpu-horovod:base

# Create the container
udocker create --name=multigpu-base gonzabad/multigpu-horovod:base
```

This image handles OpenMPI and Horovod installation taking into account the OpenMPI version available in the cluster, so we need to pass it as argument to the `--env` parameter when running the container for the first time:

```
# Add the GPU libraries to the container
udocker setup --nvidia --force multigpu-base

# Run it and install OpenMPI and Horovod
udocker run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-base
 ```

It is important to execute these last steps on the GPU nodes on which we will train our deep learning models to ensure the full compatibility with the NVIDIA drivers. All the required steps for the proper installation can also be found in [setup-container.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/setup-container.sh) script.

Once we have set up the container we can train the models. The training can be launched with the [control.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/control.sh) script. This script controls aspects like the name of the container to execute:

```
export CONTAINER="multigpu-base"
```

the job to run (in this case we will run the one corresponding to udocker):

```
export JOB2RUN="./job_udocker.sh" # Select job script based on type of execution (udocker vs native)
```

which host directory mount into the container (this directory must include the [benchmark script](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/benchmark/tensorflow2_synthetic_benchmark.py)):

```
export DIR_TO_MOUNT="$HOME/experiments/synthetic-benchmark"
```

and the variables controlling the GPU configurations on which the model will be trained, by default each model will be trained in six different configurations spanning from 1 to 6 GPUs:

```
export NUM_GPUS=('1' '2' '3' '4' '5' '6')
```

Differents HPC cluster could have different Slurm directives, so please take this into account.

This script launches the [job_udocker.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/job_udocker.sh) script which finally sets up the MPI parameters and run the jobs via udocker.

Once the [control.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/control.sh) is launched via some workload manager (Slurm in this case) the distributed training of the deep learning models will be executed. When it finish Slurm will return a log file with the images/sec processed for each model on each GPU configuration, besides a .pkl file containing these results. This last file will be saved in the [results](https://github.com/jgonzalezab/workflow-DL-HPC/tree/main/experiments/synthetic-benchmark/benchmark/results) folder. 

#### Native environment

To ensure proper scalability of the containerized workflow, we compared it to a native version with no virtualization involved. This installation is highly dependent on the specific HPC configuration and the version of the software available on it. For such purpose we created a Conda environment following the [instructions provided by Horovod](https://horovod.readthedocs.io/en/stable/conda_include.html). Note that during this installation it is important to be consistent with the versions of the software used in the Container Based environment so that the comparison of results is fair.

The script control.sh is also used to launch jobs in this environment. Users just need to make sure of running the job_native.sh script:

```
export JOB2RUN="./job_native.sh"
```
and specify the correct conda environment:

```
export ENV_NAME="multiGPU"
```
Once this script is launched and training is completed, it will return the results through Slurm log files.

### Statistical Downscaling

This experiment performs Statistical Downscaling of precipitation over the region of North America via Deep Learning. The objective of this experiment is to prove that the proposed workflow can be useful in a real scientific case, as well as show its adapatibility to different HPC clusters.

#### Container set-up

The [Dockerfile](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/Dockerfile) developed in this experiment is the result of adding the specific software needed to run the Statistical Downscaling to the Dockerfile of the TensorFlow benchmark. The image can be accessed through [DockerHub](https://hub.docker.com/layers/gonzabad/multigpu-horovod/downscaling/images/sha256-d695141efc6677e0ac38e6702c5a0c580ed8bc7613a6e5063e609c3e83a738b1?context=explore). The container can be configured in the same way as in the previous experiment:

```
# Pull image fom DockerHub
udocker pull gonzabad/multigpu-horovod:downscaling

# Create the container
udocker create --name=multigpu-downscaling gonzabad/multigpu-horovod:downscaling

# Add the GPU libraries to the container
udocker setup --nvidia --force multigpu-downscaling

# Run it and install OpenMPI and Horovod
udocker run --env="OpenMPI=4.1.0" --env="HOROVOD=latest" multigpu-downscaling
```

All these step can also be found on the [setup-container.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/setup-container.sh) script.

#### Download and preprocess data

In order to train the model we need to download and preprocess the required data. A [download_data.R](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/precipitation/download_data.R) script is provided for downloading the data from the [User Data Getaway - Thredds Access Portal (UDG-TAP)](http://meteo.unican.es/udg-tap/home) (an account may be needed). To download the data just run:

```
Rscript download_data.R
```

It will save the download data in the `/precipitation/data/` folder. Both datasets ([ERA-Interim](http://dx.doi.org/10.21957/vf291hehd7) and [EWEMBI](http://doi.org/10.5880/pik.2019.004)) can also be downloaded from their original sources. Now the data can be preprocessed wit the [preprocess.R script](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/precipitation/preprocess.R):

```
Rscript preprocess.R
```

This will save an `rda` in the `data` folder with all the data needed to train the models.

#### Train the models

Once the data is ready we can train the model. The previous step was done using R given the use of climate4R but the distributed training of the model we will done using Python. The execution of the experiments can be controlled with the [statistical-downscaling-forhlr2.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/statistical-downscaling-forhlr2.sh) script in the same way as in the TensorFlow benchmark. We train the model on four different configurations spanning from 1 to 4 GPUs:

```
export NUM_NODES=1
export NTASKS=('1' '2' '3' '4')
```

It launchs the [SD-job.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/precipitation/SD-job.sh) script which sets up the MPI parameters and run the jobs via udocker. The Slurm directives used in this experiment are specific of the ForHLR2 cluster.

Results of the experiment will be returned via Slurm log files.
