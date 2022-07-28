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

srun -c2 --gres=gpu:$NUM_GPUS <<EOF
#!/bin/bash
echo "*** SLURM RUN STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF
