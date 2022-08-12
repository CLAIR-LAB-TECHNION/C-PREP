#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$(dirname $SCRIPT_DIR)

nohup python $SCRIPT_DIR/experiments_multi_sbatch.py $@ > multi_sbatch.out &

