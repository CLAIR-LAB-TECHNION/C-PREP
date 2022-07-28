#!/bin/bash
###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cuda

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# check cuda version
/usr/local/cuda/bin/nvcc --version

# Check cuda working with pytorch
python cuda_check.py

