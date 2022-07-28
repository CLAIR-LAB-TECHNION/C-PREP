#!/bin/bash


nohup srun -c2 --gres=gpu:1 jupyter-lab.sh > jlab.out &
