#!/bin/bash

args=(
    GN-4X4-1PAS-DQN  # job name
    newton9  # server name
    2  # num cpus
    0  # num gpus
    --num_workers 128  # number of maximum jobs to run simultaneously
    --num_seeds 5  # number of times to repeat each experiment
    --experiment WithTransferExperiment  # experiment type
    --env gn_4x4_1pas  # env setting
    --context fixed_entities  # context space
    --num_src_samples 1 3 5 10  # context samples
    --num_tgt_samples 1 3 5 10
    --use_tgt_for_test  # test generalization
    # --mods                # MF
    --mods OHE            # OHE uninformative context
    --mods HCV            # HCV informative context
    # --mods AS             # RM abstract state
    # --mods RS             # RM reward shaping
    # --mods AS RS          # Camacho
    # --mods AS OHE         # AS + reps
    # --mods AS HCV
    # --mods RS OHE         # RS + reps
    # --mods RS HCV
    --mods AS RS OHE      # Camacho + context rep
    --mods AS RS HCV
    --mods AS NDS RS      # Ours!
    --mods AS NDS RS OHE  # Ours! + context reps
    --mods AS NDS RS HCV
    # --mods AS NDS         # ablations
    # --mods NDS RS
    # --mods NDS
    # --mods AS NDS OHE      # ablations + context reps
    # --mods NDS RS OHE
    # --mods NDS OHE
    # --mods AS NDS HCV
    # --mods NDS RS HCV
    # --mods NDS  HCV
    --grid_resolution 1 1  # grid resolutions for RM
    --grid_resolution 2 2
    --grid_resolution 4 4
    --goal_state_reward 100  # other RM hyperparameters
    --rs_gamma 0.999
    --alg DQN  # algorithm 
#    --dqn_target_update_interval 1000
    --timesteps 1e6  # time dependent arguments
    --dqn_exploration_timesteps 2e5
#    --min_timesteps 8e5
#    --max_no_improvement_evals 10
    --eval_freq 1e4
    --n_eval_episodes 20
    --chkp_freq 2e5
)


scripts/run_multi_sbatch_in_background.sh "${args[@]}"