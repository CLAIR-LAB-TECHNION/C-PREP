import argparse
from collections import OrderedDict
from collections.abc import Iterable as IterableType
import pprint
import time
from itertools import product
from tqdm.auto import tqdm
import math

from rmrl.experiments.configurations import *
from rmrl.experiments.runner import ExperimentsRunner
from rmrl.nn import models
from rmrl.utils.misc import powerset

EXP_CHOICES = [exp_label.value for exp_label in SupportedExperiments]
ENV_CHOICES = [env_label.value for env_label in SupportedEnvironments]
CONTEXT_CHOICES = [context_label.value for context_label in ContextSpaces]


def main():
    args = parse_args()
    print('command line arguments:')
    pprint.PrettyPrinter().pprint(vars(args))
    print()

    # set up experiment configurations
    print('collecting experiment configurations')
    start = time.time()
    single_run_args_list = get_single_run_args_list(args)
    cfgs = get_all_configurations(single_run_args_list)
    end = time.time()
    print(f'collect configurations execution time {end - start}\n')

    # run all experiments
    runner = ExperimentsRunner(args.experiment, cfgs, args.timesteps, args.log_interval, args.n_eval_episodes,
                               args.eval_freq, args.max_no_improvement_evals, args.min_evals, args.sample_seed,
                               args.num_src_samples, args.num_tgt_samples, args.num_workers, args.verbose)
    print(f'running {runner.num_runs} experiments')
    if args.count_only:
        exit()

    start = time.time()
    runner.run()
    end = time.time()
    print(f'all experiments execution time {end - start}\n')


def get_all_configurations(single_run_args_list):
    unique_cfgs_map = {}
    for run_args in single_run_args_list:
        rm_kwargs = dict(
            goal_state_reward=run_args.goal_state_reward,
            grid_resolution=run_args.grid_resolution,
            fuel_resolution=run_args.fuel_resolution
        )
        model_kwargs = dict(
            ofe_hidden_dims=run_args.ofe_hidden_dims,
            ofe_out_dim=run_args.ofe_out_dim,
            gnn_hidden_dims=run_args.gnn_hidden_dims,
            gnn_out_dim=run_args.gnn_out_dim,
            gnn_agg=run_args.gnn_agg
        )
        alg_kwargs = dict(
            learning_rate=run_args.learning_rate,
            batch_size=run_args.batch_size,
            gamma=run_args.gamma
        )

        # alg-specific kwargs
        if 'off_policy_learning_starts' in run_args:
            alg_kwargs['learning_starts'] = run_args.off_policy_learning_starts
        if 'off_policy_train_freq' in run_args:
            alg_kwargs['train_freq'] = run_args.off_policy_train_freq
        if 'off_policy_train_freq_episodes' in run_args:
            alg_kwargs['train_freq_episodes'] = run_args.off_policy_train_freq_episodes
        if 'off_policy_gradient_steps' in run_args:
            alg_kwargs['gradient_steps'] = run_args.off_policy_gradient_steps
        if 'on_policy_n_steps' in run_args:
            alg_kwargs['n_steps'] = run_args.on_policy_n_steps
        if 'dqn_exploration_fraction' in run_args:
            alg_kwargs['exploration_fraction'] = run_args.dqn_exploration_fraction

        cfg = ExperimentConfiguration(
            env=run_args.env,
            cspace=run_args.context,
            seed=run_args.seed,
            alg=run_args.alg,
            mods=run_args.mods,
            rm_kwargs=rm_kwargs,
            model_kwargs=model_kwargs,
            alg_kwargs=alg_kwargs
        )

        if repr(cfg) not in unique_cfgs_map:
            unique_cfgs_map[repr(cfg)] = cfg

    return list(unique_cfgs_map.values())

def get_single_run_args_list(args):
    args_dict = vars(args)
    iterable_args_dict = OrderedDict(filter(lambda kv: isinstance(kv[1], IterableType), args_dict.items()))
    single_value_args_dict = {k: v for k, v in args_dict.items() if k not in iterable_args_dict}

    single_run_args_dict_items_set = set()
    for subset_v in tqdm(product(*iterable_args_dict.values()), total=math.prod(map(len, iterable_args_dict.values()))):
        d = dict(zip(iterable_args_dict.keys(), subset_v))
        d.update(single_value_args_dict)

        alg = d['alg']
        if alg not in OFF_POLICY_ALGOS:
            d.pop('off_policy_learning_starts')
            d.pop('off_policy_train_freq')
            d.pop('off_policy_train_freq_episodes')
            d.pop('off_policy_gradient_steps')
        elif alg not in ON_POLICY_ALGOS:
            d.pop('on_policy_n_steps')
        if alg != Algos.DQN:
            d.pop('dqn_exploration_fraction')

        hashable_d = tuple((k, tuple(v)) if isinstance(v, Iterable) else (k, v) for k, v in d.items())
        if hashable_d not in single_run_args_dict_items_set:
            single_run_args_dict_items_set.add(hashable_d)

    return [argparse.Namespace(**dict(d_items)) for d_items in single_run_args_dict_items_set]


def parse_args():
    parser = argparse.ArgumentParser(description='Run RMRL experiemnts')

    # general experiment args
    exp_group = parser.add_argument_group('experiment values')
    exp_group.add_argument('--count_only',
                           help='only display the number of experiments slotted to run, then exit',
                           action='store_true')
    exp_group.add_argument('--experiment',
                           help='the experiment type to run',
                           choices=SupportedExperiments,
                           nargs='*',
                           type=SupportedExperiments,
                           default=list(SupportedExperiments))
    exp_group.add_argument('--env',
                           help='the environment on which to experiment',
                           choices=SupportedEnvironments,
                           nargs='*',
                           type=SupportedEnvironments,
                           default=list(SupportedEnvironments))
    exp_group.add_argument('--context',
                           help='context to be tested in compatible env',
                           choices=ContextSpaces,
                           nargs='*',
                           type=ContextSpaces,
                           default=list(ContextSpaces))
    exp_group.add_argument('--seed',
                           help='random seed for experiment',
                           type=int,
                           nargs='*',
                           default=SEEDS)
    exp_group.add_argument('--sample_seed',
                           help='random seed for context sampling',
                           type=int,
                           nargs='*',
                           default=SAMPLE_SEEDS)
    exp_group.add_argument('--num_src_samples',
                           help='the number of samples in the source contexts set',
                           type=int,
                           nargs='*',
                           default=NUM_SRC_SAMPLES)
    exp_group.add_argument('--num_tgt_samples',
                           help='the number of samples in the target contexts set',
                           type=int,
                           nargs='*',
                           default=NUM_TGT_SAMPLES)

    # policy config args
    policy_group = parser.add_argument_group('policy configurations')
    policy_group.add_argument('--alg',
                              help='the underlying RL algorithm',
                              choices=Algos,
                              type=Algos,
                              nargs='*',
                              default=list(Algos))
    policy_group.add_argument('--mods',
                              help='if true, will include abstract state in model input',
                              action='append',
                              choices=Mods,
                              type=Mods,
                              # default=list(powerset(Mods)),
                              nargs='*')

    # learning args
    learning_group = parser.add_argument_group('learning configurations')
    learning_group.add_argument('--learning_rate',
                                help='model learning rate',
                                type=float,
                                nargs='*',
                                default=LEARNING_RATES)
    learning_group.add_argument('--batch_size',
                                help='training mini-batch size',
                                type=int,
                                nargs='*',
                                default=BATCH_SIZES)
    learning_group.add_argument('--gamma',
                                help='discount factor for chosen RL algo',
                                type=float,
                                nargs='*',
                                default=GAMMA)
    learning_group.add_argument('--timesteps',
                                help='max number of timesteps for the trained agent',
                                type=lambda x: int(float(x)),
                                default=TOTAL_TIMESTEPS)

    # algo specific
    learning_group.add_argument('--on_policy_n_steps',
                                help='for on-policy algorithms only! number of steps per experience rollout',
                                type=lambda x: int(float(x)),
                                nargs='*',
                                default=ON_POLICY_N_STEPS)
    learning_group.add_argument('--off_policy_learning_starts',
                                help='for off-policy algorithms only! minimal number of steps to take before learning',
                                type=lambda x: int(float(x)),
                                nargs='*',
                                default=OFF_POLICY_LEARNING_STARTS)
    learning_group.add_argument('--off_policy_train_freq',
                                help='for off-policy algorithms only! number of steps per rollout',
                                type=lambda x: int(float(x)),
                                nargs='*',
                                default=OFF_POLICY_TRAIN_FREQ)
    learning_group.add_argument('--off_policy_train_freq_episodes',
                                help='for off-policy algorithms only! change `train_freq` to be measured in episodes',
                                action='store_true')
    learning_group.add_argument('--off_policy_gradient_steps',
                                help='for on-policy algorithms only! number of learning steps per rollout. -1 == all',
                                type=lambda x: int(float(x)),
                                nargs='*',
                                default=OFF_POLICY_GRADIENT_STEPS)
    learning_group.add_argument('--dqn_exploration_fraction',
                                help='for DQN only! fraction of training using declining epsilon greedy',
                                type=float,
                                nargs='*',
                                default=DQN_EXPLORATION_FRACTIONS)

    # RM params
    # TODO make more general
    rm_group = parser.add_argument_group('RM configurations')
    rm_group.add_argument('--goal_state_reward',
                          help='RM reward when reaching the goal state',
                          type=float,
                          nargs='*',
                          default=GOAL_STATE_REWARDS)
    rm_group.add_argument('--grid_resolution',
                          help='number of cells per grid sector',
                          nargs=2,
                          type=int,
                          # default=GRID_RESOLUTIONS,
                          action='append')
    rm_group.add_argument('--fuel_resolution',
                          help='number of fuel waypoints',
                          type=int,
                          nargs='*',
                          default=FUEL_RESOLUTIONS)

    # RM feature extractor params
    model_group = parser.add_argument_group('NN model configurations')
    model_group.add_argument('--ofe_hidden_dims',
                             help='number of hidden features in the layers of the observation feature extractor MLP',
                             nargs='*',
                             type=int,
                             # default=HIDDEN_DIMS,
                             action='append')
    model_group.add_argument('--ofe_out_dim',
                             help='number of output features of the observation feature extractor MLP',
                             type=int,
                             default=OUT_DIMS,
                             nargs='*')
    model_group.add_argument('--gnn_hidden_dims',
                             help='number of hidden features in the layers of the RM GNN',
                             nargs='*',
                             type=int,
                             # default=HIDDEN_DIMS,
                             action='append')
    model_group.add_argument('--gnn_out_dim',
                             help='number of output features of the RM GNN',
                             type=int,
                             default=OUT_DIMS,
                             nargs='*')
    model_group.add_argument('--gnn_agg',
                             help='the type of aggregation to perform on RM node embeddings',
                             choices=NODE_AGGS,
                             type=lambda x: getattr(models, x),
                             nargs='*',
                             default=NODE_AGGS)

    # logging params
    log_group = parser.add_argument_group('logging configurations')
    log_group.add_argument('--verbose',
                           help='display training outputs',
                           action='store_true')
    log_group.add_argument('--log_interval',
                           help='number of training iterations / episodes per info dump',
                           type=int,
                           default=LOG_INTERVAL)
    log_group.add_argument('--n_eval_episodes',
                           help='number of episodes to run per evaluation',
                           type=int,
                           default=N_EVAL_EPISODES)
    log_group.add_argument('--eval_freq',
                           help='number of timesteps per evaluation',
                           type=int,
                           default=EVAL_FREQ)
    log_group.add_argument('--max_no_improvement_evals',
                           help='training will stop if no improvement is seen for this many evaluations for early '
                                'stopping',
                           type=int,
                           default=MAX_NO_IMPROVEMENT_EVALS)
    log_group.add_argument('--min_evals',
                           help='minimal number of evaluations that must occur, regardless of the '
                                '--max_no_improvement_evals argument',
                           type=int,
                           default=MIN_EVALS)

    # technical params
    tech_group = parser.add_argument_group('technical configurations')
    tech_group.add_argument('--num_workers',
                            help='max number of threads for running experiments in parallel',
                            type=int,
                            default=1)

    args = parser.parse_args()

    # handle hidden dim values with 'append' action issue using default value
    if args.mods is None:
        args.mods = list(powerset(Mods))
    if args.grid_resolution is None:
        args.grid_resolution = GRID_RESOLUTIONS
    if args.ofe_hidden_dims is None:
        args.ofe_hidden_dims = HIDDEN_DIMS
    if args.gnn_hidden_dims is None:
        args.gnn_hidden_dims = HIDDEN_DIMS

    return args


if __name__ == '__main__':
    main()
