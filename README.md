# A Container Based Workflow for Distributed Training of Deep Learning Algorithms in HPC Clusters

This repository includes all the code to reproduce the results of the paper *A Container Based Workflow for Distribute Training of Deep Learning Algorithms in HPC Clusters*

* Experiments: This folder contains the actual experiments
	* **Synthetic benchmark**: Train InceptionV3, ResNet50 and ResNet101 over synthetic data using [the script provided by Horovod](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_synthetic_benchmark.py). The steps to run this experiment are the following:
		1. Install udocker in the HPC cluster (see [installation manual](https://indigo-dc.gitbook.io/udocker/installation_manual))
		2. Run [setup-container.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/setup-container.sh) for creating the container (taking into account the OpenMPI version available in the cluster)
		3. Run [control.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/control.sh) specifing the jobs to run through its parameters

	The image was build from the [Dockerfile](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/Dockerfile) here included. Specific details of the udocker and native jobs can be found in [job_udocker.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/job_udocker.sh) and [job_native.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/job_native.sh). The native environment was setup following [setup-native-altamira.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/synthetic-benchmark/setup-native-altamira.sh)

	* **Statistical Downscaling**: Train a fully convolutional model to the task of Statistical Downscaling over the region of North America. The steps to run this experiment are the following:
		1. Install udocker in the HPC cluster
		2. Run [setup-container.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/setup-container.sh) for creating the container (taking into account the OpenMPI version available in the cluster)
		3. Download the data from [UDG](http://meteo.unican.es/udg-tap/home) by running [download_data.R](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/precipitation/download_data.R)
		4. Run [statistical-downscaling-forhlr2.sh](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/statistical-downscaling-forhlr2.sh) specifing the jobs to run through its parameters

	The image was build from the [Dockerfile](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/Dockerfile) here included. The [precipitation folder](https://github.com/jgonzalezab/workflow-DL-HPC/blob/main/experiments/statistical-downscaling/precipitation) contains the code for downloading and preprocessing the data, the model definiton and its training. For downloading the data an user in [UDG](http://meteo.unican.es/udg-tap/home) is required, although the raw data can be downloaded from [ERA-Interim](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim) and [EWEMBI](https://dataservices.gfz-potsdam.de/pik/showshort.php?id=escidoc:3928916) official sites. Further details on the job config can be found in SD-job.sh


