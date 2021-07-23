# Base image setting
ARG image=gonzabad/multigpu-horovod
ARG tag=base

FROM ${image}:${tag}

####### Intall custom libraries below #######

# Install R from r-project repos
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    	software-properties-common=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    	r-base=\* \
        r-base-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install R and Java tools
RUN apt-get clean && apt-get autoclean

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
	 libgit2-dev=\* \
         libcurl4-gnutls-dev=\* \
         libicu-dev=\* \
         libbz2-dev=\* \
         libnetcdf-dev=\* \
         libnetcdff-dev=\* \
	 libssl-dev=\* \
         libssh2-1-dev=\* \
         libxml2-dev=\* \
         libgit2-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN R -e "install.packages('devtools')"
    
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
         default-jre=\* \
         default-jdk=\* && \
    R CMD javareconf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl repo.data.kit.edu/key.pgp | apt-key add - && \
    add-apt-repository "deb http://repo.data.kit.edu/ubuntu/$(lsb_release -sr) ./" && \
    apt-get install -y --no-install-recommends \
    	liblzma-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install rpy2
RUN pip install --no-cache-dir rpy2

# Install Keras
RUN pip install --no-cache-dir keras h5py

# Install virtual framebuffer X11 server 
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
		xvfb=\* \
        xauth=\* \
        xfonts-base=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install climate4R environment (R libraries) 
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR.java')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/climate4R.UDG')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/transformeR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/downscaleR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/VALUE')"
RUN R -e "library('devtools'); install_github('SantanderMetGRoup/climate4R.value')"
RUN R -e "library('devtools'); install_github('SantanderMetGRoup/downscaleR.keras@devel')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR.2nc')"
RUN R -e "install.packages(c('Matrix', 'RcppEigen', 'spam'))"

# Install Keras and Tensorflow R API
RUN R -e "install.packages('keras')"
RUN R -e "reticulate::use_python('/usr/local/bin/python', required = TRUE)"
