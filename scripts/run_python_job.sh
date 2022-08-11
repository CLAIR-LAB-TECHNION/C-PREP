#!/bin/bash

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=$2
NUM_GPUS=$3
JOB_NAME=$1
MAIL_USER="guy.azran@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# move num gpus argument
shift 3

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=rmrl

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-${JOB_NAME}-%N-%j.out' \
	-e 'slurm-${JOB_NAME}-%N-%j.err' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

