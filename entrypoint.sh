#!/bin/bash

# Source the conda.sh script to make `conda` command available.
# This path is standard for Miniconda installations. Adjust if different.
source /opt/conda/etc/profile.d/conda.sh

# Get the desired default Conda environment name from an environment variable.
# If DEFAULT_CONDA_ENV is not set, it defaults to 'base'.
DEFAULT_ENV="${DEFAULT_CONDA_ENV:-base}"

# Activate the default environment for the container's primary execution.
# This ensures that `run.py` or `singularity shell` starts in the specified default env.
echo "Activating default Conda environment: $DEFAULT_ENV"
conda activate "$DEFAULT_ENV"

# Check if activation was successful (optional, but good for debugging)
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate default Conda environment '$DEFAULT_ENV'."
    echo "Available environments:"
    conda env list
    exit 1
fi

# Execute the command passed to the container (e.g., '/app/run.py', 'python', 'bash', etc.)
# This ensures that any command you run with `docker run` or `singularity run/exec`
# is executed within the activated default Conda environment.
exec "$@"