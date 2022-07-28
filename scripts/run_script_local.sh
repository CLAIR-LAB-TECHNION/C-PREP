#!/bin/bash

###
# Parameters for srun
#
NUM_GPUS=2

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=rmrl

srun -c2 --gres=gpu:$NUM_GPUS "$1"
