import argparse
import pprint
import time
from itertools import product

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
    cfgs = get_all_configurations(args)
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


def get_all_configurations(args):
    cfgs = []

    for (
            env,
            context,
            seed,
            alg,
            mods,
            learning_rate,
            batch_size,
            goal_state_reward,
            grid_resolution,
            fuel_resolution,
            ofe_hidden_dims,
            ofe_out_dim,
            gnn_hidden_dims,
            gnn_out_dim,
            gnn_agg,
    ) in iterate_arg_combinations(args):
        # prepare special kwargs
        rm_kwargs = dict(
            goal_state_reward=goal_state_reward,
            grid_resolution=grid_resolution,
            fuel_resolution=fuel_resolution
        )
        model_kwargs = dict(
            ofe_hidden_dims=ofe_hidden_dims,
            ofe_out_dim=ofe_out_dim,
            gnn_hidden_dims=gnn_hidden_dims,
            gnn_out_dim=gnn_out_dim,
            gnn_agg=gnn_agg
        )
        alg_kwargs = dict(
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        # handle algorithm-specific kwargs
        alg_specific_kwargs_list = []
        if alg in OFF_POLICY_ALGOS:
            for learning_starts in args.off_policy_learning_starts:
                for train_freq in args.off_policy_train_freq:
                    if args.off_policy_train_freq_episodes:
                        train_freq = (train_freq, 'episode')
                    for gradient_steps in args.off_policy_gradient_steps:
                        if alg == Algos.DQN:
                            for exploration_fraction in args.dqn_exploration_fraction:
                                specific_kwargs = alg_kwargs.copy()
                                specific_kwargs.update(dict(
                                    learning_starts=learning_starts,
                                    train_freq=train_freq,
                                    gradient_steps=gradient_steps,
                                    exploration_fraction=exploration_fraction
                                ))
                                alg_specific_kwargs_list.append(specific_kwargs)
                        else:
                            specific_kwargs = alg_kwargs.copy()
                            specific_kwargs.update(dict(
                                learning_starts=learning_starts,
                                train_freq=train_freq,
                                gradient_steps=gradient_steps
                            ))
                            alg_specific_kwargs_list.append(specific_kwargs)
        elif alg in ON_POLICY_ALGOS:
            for n_steps in args.on_policy_n_steps:
                specific_kwargs = alg_kwargs.copy()
                specific_kwargs.update(dict(
                    n_steps=n_steps
                ))
                alg_specific_kwargs_list.append(specific_kwargs)

        cfgs.extend([
            ExperimentConfiguration(
                env=env,
                cspace=context,
                seed=seed,
                alg=alg,
                mods=mods,
                rm_kwargs=rm_kwargs,
                model_kwargs=model_kwargs,
                alg_kwargs=alg_specific_kwargs
            )
            for alg_specific_kwargs in alg_specific_kwargs_list
        ])

    return cfgs


def iterate_arg_combinations(args):
    return product(
        args.env,
        args.context,
        args.seed,
        args.alg,
        args.mods,
        args.learning_rate,
        args.batch_size,
        args.goal_state_reward,
        args.grid_resolution,
        args.fuel_resolution,
        args.ofe_hidden_dims,
        args.ofe_out_dim,
        args.gnn_hidden_dims,
        args.gnn_out_dim,
        args.gnn_agg,
    )


def get_config_count(args):
    return (
            len(args.env) *
            len(args.context) *
            len(args.seed) *
            len(args.alg) *
            len(args.mods) *
            len(args.learning_rate) *
            len(args.batch_size) *
            len(args.goal_state_reward) *
            len(args.grid_resolution) *
            len(args.fuel_resolution) *
            len(args.ofe_hidden_dims) *
            len(args.ofe_out_dim) *
            len(args.gnn_hidden_dims) *
            len(args.gnn_out_dim) *
            len(args.gnn_agg)
    )


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


# def split_group_args(parser, args):
#     group_args = []
#     for group in parser._action_groups:
#         group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
#         arg_groups[group.title] = argparse.Namespace(**group_dict)


if __name__ == '__main__':
    main()
