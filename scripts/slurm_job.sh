#!/bin/bash

###
# Parameters for sbatch
#
COMMAND=$1  # valid values are srun, sbatch
JOB_NAME=$2
NUM_NODES=1
NUM_CORES=$3
NUM_GPUS=$4
MAIL_USER="guy.azran@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# move num gpus argument
shift 4

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=rmrl

$COMMAND \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-'"${JOB_NAME}"'-%N-%j.out' \
	-e 'slurm-'"${JOB_NAME}"'-%N-%j.err' \
	$@

