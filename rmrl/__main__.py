import argparse

from rmrl.experiments.configurations import *
from rmrl.experiments.runner import ExperimentsRunner
from rmrl.nn import models
from rmrl.utils.misc import powerset
import time

EXP_CHOICES = [exp_label.value for exp_label in SupportedExperiments]
ENV_CHOICES = [env_label.value for env_label in SupportedEnvironments]
CONTEXT_CHOICES = [context_label.value for context_label in ContextSpaces]


def main():
    args = parse_args()

    # import pprint
    # pprint.PrettyPrinter().pprint(vars(args))
    # return

    # set up experiment configurations
    print('collecting experiment configurations')
    start = time.time()
    cfgs = get_all_configuraitions(args)
    end = time.time()
    print(f'collect configurations execution time {end - start}')

    # run all experiments
    print(f'running {len(args.experiment) * len(cfgs)} experiments')
    start = time.time()
    ExperimentsRunner(args.experiment, cfgs, args.sample_seed, args.num_workers, args.verbose).run()
    end = time.time()
    print(f'all experiments execution time {end - start}')

def get_all_configuraitions(args):
    cfgs = []
    for env in args.env:
        for context in args.context:
            for seed in args.seed:
                for alg in args.alg:
                    for mods_set in args.mods:
                        for learning_rate in args.learning_rate:
                            for batch_size in args.batch_size:
                                for goal_state_reward in args.goal_state_reward:
                                    for grid_resolution in args.grid_resolution:
                                        for fuel_resolution in args.fuel_resolution:
                                            for ofe_hidden_dims in args.ofe_hidden_dims:
                                                for ofe_out_dim in args.ofe_out_dim:
                                                    for gnn_hidden_dims in args.gnn_hidden_dims:
                                                        for gnn_out_dim in args.gnn_out_dim:
                                                            for gnn_agg in args.gnn_agg:
                                                                env_kwargs = RMENV_DICT[env][ENV_KWARGS_KEY]
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

                                                                # alg specific kwargs
                                                                if alg == Algos.DQN:
                                                                    for ef in args.exploration_fraction:
                                                                        alg_kwargs['exploration_fraction'] = ef
                                                                        cfgs.append(
                                                                            ExperimentConfiguration(
                                                                                env=env,
                                                                                cspace=context,
                                                                                seed=seed,
                                                                                alg=alg,
                                                                                mods=mods_set,
                                                                                env_kwargs=env_kwargs,
                                                                                rm_kwargs=rm_kwargs,
                                                                                model_kwargs=model_kwargs,
                                                                                alg_kwargs=alg_kwargs
                                                                            )
                                                                        )
                                                                else:  # no alg specific kwargs
                                                                    cfgs.append(
                                                                        ExperimentConfiguration(
                                                                            env=env,
                                                                            cspace=context,
                                                                            seed=seed,
                                                                            alg=alg,
                                                                            mods=mods_set,
                                                                            env_kwargs=env_kwargs,
                                                                            rm_kwargs=rm_kwargs,
                                                                            model_kwargs=model_kwargs,
                                                                            alg_kwargs=alg_kwargs
                                                                        )
                                                                    )
        return cfgs


def parse_args():
    parser = argparse.ArgumentParser(description='Run RMRL experiemnts')

    # general experiment args
    exp_group = parser.add_argument_group('experiment values')
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
    learning_group.add_argument('--exploration_fraction',
                                help='for DQN only! fraction of training using declining epsilon greedy',
                                type=float,
                                nargs='*',
                                default=EXPLORATION_FRACTIONS)

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

    # technical params
    tech_group = parser.add_argument_group('technical configurations')
    tech_group.add_argument('--num_workers',
                            help='max number of threads for running experiments in parallel',
                            type=int,
                            default=1)
    tech_group.add_argument('--verbose',
                            help='display training outputs',
                            action='store_true')

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
