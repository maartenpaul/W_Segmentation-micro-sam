# Choose a base image with CUDA support.
# Select a CUDA version compatible with the PyTorch needed for micro_sam and your host drivers.
# Using an Ubuntu base often works well with Conda. Using a -devel image includes compilers if needed.
# Example: CUDA 11.8 on Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# Example: CUDA 12.1 on Ubuntu 22.04
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH 

# Define default micro-sam cache directory
ENV MICROSAM_CACHEDIR=/tmp/models/microsam_cache

# Install base dependencies and cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgeos-dev \
    libgl1-mesa-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the default cache directory
RUN mkdir -p ${MICROSAM_CACHEDIR} && chmod 777 ${MICROSAM_CACHEDIR}

# ------------------------------------------------------------------------------
# Install Miniconda
# ------------------------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y && \
    # The symlink below makes conda.sh discoverable by shell startup scripts
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    # Initialize conda for bash - crucial for 'conda run' later
    /opt/conda/bin/conda init bash

# Make conda available in RUN instructions using the initialized shell
SHELL ["/bin/bash", "--login", "-c"]

# Update conda
RUN conda update -n base -c defaults conda --yes

# ------------------------------------------------------------------------------
# Create Cytomine environment (Python 3.7)
# ------------------------------------------------------------------------------
ENV CYTOMINE_ENV_NAME=cytomine_py37
RUN conda create -n $CYTOMINE_ENV_NAME python=3.7 -y

RUN conda run -n $CYTOMINE_ENV_NAME pip install --no-cache-dir \
    git+https://github.com/cytomine-uliege/Cytomine-python-client.git@v2.7.3

RUN conda run -n $CYTOMINE_ENV_NAME pip install --no-cache-dir \
    git+https://github.com/Neubias-WG5/biaflows-utilities.git@v0.9.2

RUN conda run -n $CYTOMINE_ENV_NAME pip install --no-cache-dir \
    imageio==2.9.0 \
    numpy==1.19.4 \
    numba==0.50.1 \
    cellpose==0.6.1

# ------------------------------------------------------------------------------
# Create micro_sam environment
# ------------------------------------------------------------------------------
ENV MICROSAM_ENV_NAME=microsam_env
RUN conda create -n $MICROSAM_ENV_NAME -c conda-forge micro_sam=1.5.0 -y
# OR explicit CUDA version if needed:
# RUN conda create -n $MICROSAM_ENV_NAME -c conda-forge micro_sam "pytorch=*=*cuda11.8*" -y

# Clean up conda cache
RUN conda clean -a -y

# ------------------------------------------------------------------------------
# Application Code & Entrypoint
# ------------------------------------------------------------------------------
WORKDIR /app
COPY run.py /app/run.py
COPY descriptor.json /app/descriptor.json

# This is the simplified ENTRYPOINT:
# It sources conda.sh, activates your Cytomine environment, and then runs run.py
# The "$@" ensures any arguments you pass to `docker run` are sent to run.py
ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate cytomine_py37 && exec python /app/run.py \"$@\"", "--"]

# Set a default command if no arguments are provided to `docker run`.
# If you run `docker run your_image`, it will implicitly pass "" as "$@" to the ENTRYPOINT.
# If you provide args like `docker run your_image --local`, those args become "$@".
CMD []