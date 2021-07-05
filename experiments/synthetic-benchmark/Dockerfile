# Base image setting
ARG image=tensorflow/tensorflow
ARG tag=2.2.0-gpu

FROM ${image}:${tag}

# Which branch to use when pushing https://github.com/vykozlov/horovod_entrypoint
ENV BRANCH=master

# Set NCCL version (select carefuly)
ENV NCCL_VERSION=2.7.8-1+cuda10.1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install basic libraries
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        cmake=\* \
        git=\* \
        wget=\* \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    	openssh-client=\* \
        openssh-server=\* && \
    	mkdir -p /var/run/sshd && \
    	apt-get clean && \
    	rm -rf /var/lib/apt/lists/*

# Allow OpenSSH to talk to containers without asking for confirmation
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN < /etc/ssh/ssh_config grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

WORKDIR /

# Push https://github.com/vykozlov/horovod_entrypoint@BRANCH and set the entrypoint
RUN git clone -b $BRANCH https://github.com/vykozlov/horovod_entrypoint
ENV PATH="${PATH}:/horovod_entrypoint"
ENTRYPOINT ["/horovod_entrypoint/horovod_entrypoint"]
