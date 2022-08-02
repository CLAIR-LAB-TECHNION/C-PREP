#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nohup srun -c2 --gres=gpu:0 $SCRIPT_DIR/jupyter-lab.sh > jlab.out &
