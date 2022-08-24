#!/bin/bash

###
# Parameters for sbatch
#
COMMAND=$1  # valid values are srun, sbatch
export JOB_NAME=$2
NUM_NODES=1
NODE_NAME=$3
NUM_CORES=$4
NUM_GPUS=$5
MAIL_USER="guy.azran@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# move num gpus argument
shift 5

###
# Conda parameters
#
export CONDA_HOME=$HOME/miniconda3
export CONDA_ENV=rmrl

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p slurm_logs

# include nodelist argument if specified, otherwise do automatic allocation
if [ "$NODE_NAME" = "any" ]; then
    NODES_ARG=""
else
    NODES_ARG="--nodelist $NODE_NAME"
fi

$COMMAND \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	$NODES_ARG \
	-o 'slurm_logs/'"${JOB_NAME}"'-%N-%j.out' \
	-e 'slurm_logs/'"${JOB_NAME}"'-%N-%j.err' \
	"$SCRIPT_DIR/slurm_script.sh" $@

