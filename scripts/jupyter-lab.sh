#!/bin/bash

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=rmrl

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
xvfb-run -a -s "-screen 0 1440x900x24" jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100

