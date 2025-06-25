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

# >>> ADDED: Define default micro-sam cache directory <<<
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

# >>> ADDED: Create the default cache directory <<<
# Ensure the default cache directory exists within the container
RUN mkdir -p ${MICROSAM_CACHEDIR} && chmod 777 ${MICROSAM_CACHEDIR}

# ------------------------------------------------------------------------------
# Install Miniconda
# ------------------------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y && \
    # The symlink below makes conda.sh discoverable by shell startup scripts
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# IMPORTANT: Initialize conda for bash explicitly here.
# This sets up the shell functions (like 'conda activate' and 'conda run')
# within the Docker image's filesystem, making them available in a shell
# that sources conda.sh (which our entrypoint will do).
RUN /opt/conda/bin/conda init bash

# Configure the default shell for subsequent RUN instructions.
# This ensures that subsequent RUN commands use bash with conda initialized.
SHELL ["/bin/bash", "--login", "-c"]

# Update conda
RUN conda update -n base -c defaults conda --yes

# ------------------------------------------------------------------------------
# Create Cytomine environment (Python 3.7)
# ------------------------------------------------------------------------------
ENV CYTOMINE_ENV_NAME=cytomine_py37
RUN conda create -n $CYTOMINE_ENV_NAME python=3.7 -y

# Use 'conda run -n <env_name>' for installing packages into specific environments
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

# Define the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the ENTRYPOINT to your custom script.
# This script will handle activating the correct Conda environment
# and then executing the main application script (run.py) or any other command.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the *default* Conda environment to activate when the container starts
# without specific instructions (e.g., for `singularity run` or `singularity shell`).
# This variable will be read by entrypoint.sh.
ENV DEFAULT_CONDA_ENV=$CYTOMINE_ENV_NAME

# Set a default command to run if no arguments are passed to the container.
# This will be passed to your entrypoint.sh.
CMD ["/app/run.py"]