import argparse
import math
import time
from collections import OrderedDict
from collections.abc import Iterable as IterableType
from itertools import product

import torch_geometric.nn
from tqdm.auto import tqdm

from rmrl.experiments.configurations import *
from rmrl.experiments.runner import ExperimentsRunner
from rmrl.nn import models

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
    runner = ExperimentsRunner(cfgs, args.log_interval, args.chkp_freq, args.num_workers, args.verbose,
                               args.force)
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
        exp_kwargs = __get_exp_kwargs(run_args)
        rm_kwargs = __get_rm_kwargs(run_args)
        model_kwargs = __get_model_kwargs(run_args)
        alg_kwargs = __get_alg_kwargs(run_args)
        tsf_kwargs = __get_tsf_kwargs(run_args)

        cfg = TransferConfiguration(env=run_args.env, cspace=run_args.context, alg=run_args.alg, mods=run_args.mods,
                                    exp_kwargs=exp_kwargs, rm_kwargs=rm_kwargs, model_kwargs=model_kwargs,
                                    alg_kwargs=alg_kwargs, num_src_samples=run_args.num_src_samples,
                                    num_tgt_samples=run_args.num_tgt_samples, max_timesteps=run_args.timesteps,
                                    eval_freq=run_args.eval_freq, n_eval_episodes=run_args.n_eval_episodes,
                                    max_no_improvement_evals=run_args.max_no_improvement_evals,
                                    min_timesteps=run_args.min_timesteps, seed=run_args.seed, tsf_kwargs=tsf_kwargs)

        if repr(cfg) not in unique_cfgs_map:
            unique_cfgs_map[repr(cfg)] = cfg

    return list(unique_cfgs_map.values())


def __get_exp_kwargs(run_args):
    exp_kwargs = dict(
        use_tgt_for_test=run_args.use_tgt_for_test,
    )

    return exp_kwargs


def __get_rm_kwargs(run_args):
    if Mods.has_rm_mod(run_args.mods):
        rm_kwargs = dict(
            rs_gamma=run_args.rs_gamma,
            goal_state_reward=run_args.goal_state_reward,
            grid_resolution=run_args.grid_resolution,
            fuel_resolution=run_args.fuel_resolution,
        )
    else:
        rm_kwargs = {}

    return rm_kwargs


def __get_model_kwargs(run_args):
    model_kwargs = {}

    if not run_args.ofe_identity:
        model_kwargs.update(dict(
            ofe_hidden_dims=run_args.ofe_hidden_dims,
            ofe_out_dim=run_args.ofe_out_dim,
        ))
    else:
        model_kwargs.update(dict(
            ofe_identity=True,
        ))
    if Mods.GECO in run_args.mods or Mods.GECOUPT in run_args.mods:
        model_kwargs.update(dict(
            gnn_type=run_args.gnn_type,
            gnn_hidden_dims=run_args.gnn_hidden_dims,
            gnn_out_dim=run_args.gnn_out_dim,
            gnn_agg=run_args.gnn_agg
        ))

    return model_kwargs


def __get_alg_kwargs(run_args):
    alg_kwargs = dict(
        learning_rate=run_args.learning_rate,
        gamma=run_args.gamma
    )

    # alg-specific kwargs
    if run_args.alg != Algos.A2C:
        alg_kwargs['batch_size'] = run_args.batch_size
    if 'on_policy_n_steps' in run_args:
        alg_kwargs['n_steps'] = run_args.on_policy_n_steps
    if 'on_policy_ent_coef' in run_args:
        alg_kwargs['ent_coef'] = run_args.on_policy_ent_coef
    if 'ppo_n_epochs' in run_args:
        alg_kwargs['n_epochs'] = run_args.ppo_n_epochs
    if 'off_policy_learning_starts' in run_args:
        alg_kwargs['learning_starts'] = run_args.off_policy_learning_starts
    if 'off_policy_train_freq' in run_args:
        alg_kwargs['train_freq'] = run_args.off_policy_train_freq
    if 'off_policy_train_freq_episodes' in run_args and run_args.off_policy_train_freq_episodes:
        alg_kwargs['train_freq'] = (run_args.off_policy_train_freq, 'episode')
    if 'off_policy_gradient_steps' in run_args:
        alg_kwargs['gradient_steps'] = run_args.off_policy_gradient_steps
    if 'dqn_exploration_timesteps' in run_args:
        alg_kwargs['exploration_timesteps'] = run_args.dqn_exploration_timesteps
    if 'dqn_target_update_interval' in run_args:
        alg_kwargs['target_update_interval'] = run_args.dqn_target_update_interval

    return alg_kwargs


def __get_tsf_kwargs(run_args):
    tsf_kwargs = dict(no_transfer=run_args.no_transfer)

    if not run_args.no_transfer:
        tsf_kwargs.update(dict(
            transfer_model=run_args.transfer_model,
            keep_timesteps=run_args.keep_timesteps,
            target_timesteps=run_args.target_timesteps,
            target_eval_freq=run_args.target_eval_freq,
        ))

        if run_args.alg in OFF_POLICY_ALGOS:
            tsf_kwargs['keep_buffer'] = run_args.keep_buffer

    return tsf_kwargs


def get_single_run_args_list(args):
    args_dict = vars(args)
    iterable_args_dict = OrderedDict(filter(lambda kv: isinstance(kv[1], IterableType), args_dict.items()))
    single_value_args_dict = {k: v for k, v in args_dict.items() if k not in iterable_args_dict}

    single_run_args_dict_items_set = set()
    for subset_v in tqdm(product(*iterable_args_dict.values()), total=math.prod(map(len, iterable_args_dict.values())),
                         desc='collecting configurations'):
        d = dict(zip(iterable_args_dict.keys(), subset_v))
        d.update(single_value_args_dict)

        # exp-specific args
        if d['no_transfer']:
            d.pop('transfer_model')
            d.pop('keep_timesteps')
            d.pop('target_timesteps')
            d.pop('target_eval_freq')

            if d['alg'] in OFF_POLICY_ALGOS:
                d.pop('keep_buffer')

        # model-specific args
        if d['ofe_identity']:
            d.pop('ofe_hidden_dims')
            d.pop('ofe_out_dim')
        if Mods.GECO not in d['mods'] and Mods.GECOUPT not in d['mods']:
            d.pop('gnn_type')
            d.pop('gnn_hidden_dims')
            d.pop('gnn_out_dim')
            d.pop('gnn_agg')

        # algorithm-specific args
        alg = d['alg']
        if alg not in OFF_POLICY_ALGOS:
            d.pop('off_policy_learning_starts')
            d.pop('off_policy_train_freq')
            d.pop('off_policy_train_freq_episodes')
            d.pop('off_policy_gradient_steps')
        elif alg not in ON_POLICY_ALGOS:
            d.pop('on_policy_n_steps')
            d.pop('on_policy_ent_coef')
        if alg != Algos.DQN:
            d.pop('dqn_exploration_timesteps')
            d.pop('dqn_target_update_interval')
        if alg != Algos.PPO:
            d.pop('ppo_n_epochs')

        hashable_d = tuple((k, tuple(v))
                           if isinstance(v, IterableType) and not isinstance(v, str)
                           else (k, v) for k, v in d.items())
        if hashable_d not in single_run_args_dict_items_set:
            single_run_args_dict_items_set.add(hashable_d)

    return [argparse.Namespace(**dict(d_items)) for d_items in single_run_args_dict_items_set]


def parse_args():
    parser = argparse.ArgumentParser(description='RMRL experiment arguments')

    # general experiment args
    exp_group = parser.add_argument_group('experiment values')
    exp_group.add_argument('--count_only',
                           help='only display the number of experiments slotted to run, then exit',
                           action='store_true')
    exp_group.add_argument('--no_transfer',
                           help='if provided, the agent will not be trained on the target contexts',
                           action='store_true')
    exp_group.add_argument('--env',
                           help='the environment on which to experiment',
                           choices=SupportedEnvironments,
                           nargs='+',
                           type=SupportedEnvironments,
                           default=list(SupportedEnvironments))
    exp_group.add_argument('--context',
                           help='context to be tested in compatible env',
                           choices=ContextSpaces,
                           nargs='+',
                           type=ContextSpaces,
                           default=list(ContextSpaces))
    exp_group.add_argument('--num_src_samples',
                           help='the number of samples in the source contexts set',
                           type=int,
                           nargs='+',
                           default=NUM_SRC_SAMPLES)
    exp_group.add_argument('--num_tgt_samples',
                           help='the number of samples in the target contexts set',
                           type=int,
                           nargs='+',
                           default=NUM_TGT_SAMPLES)
    exp_group.add_argument('--num_seeds',
                           help='number of different random seed runs',
                           type=int)
    exp_group.add_argument('--seed',
                           help='random seed for experiment',
                           type=int,
                           nargs='+')

    # experiment specific args
    # no transfer exp
    notsf_group = parser.add_argument_group('no transfer experiment values')
    notsf_group.add_argument('--use_tgt_for_test',
                             help='for NoTransferExperiment only! uses target contexts for eval during training',
                             action='store_true')

    # with transfer exp
    tsf_group = parser.add_argument_group('transfer experiment values')
    tsf_group.add_argument('--transfer_model',
                           help='for WithTransferExperiment only! which model to transfer',
                           choices=[FINAL_MODEL_NAME, BEST_MODEL_NAME],
                           nargs='+',
                           default=[BEST_MODEL_NAME])
    tsf_group.add_argument('--keep_timesteps',
                           help='for WithTransferExperiment only! if True, training starts from timestep at the end'
                                'of the transferred model training',
                           action='store_true')
    tsf_group.add_argument('--target_timesteps',
                           help='for WithTransferExperiment only! max number of timesteps for the trained agent in the '
                                'target context',
                           type=lambda x: int(float(x)),
                           default=TARGET_TIMESTEPS)
    tsf_group.add_argument('--target_eval_freq',
                           help='for WithTransferExperiment only! number of timesteps per evaluation in the target '
                                'context',
                           type=lambda x: int(float(x)),
                           default=TARGET_EVAL_FREQ)
    tsf_group.add_argument('--keep_buffer',
                           help='if True, transfers the rollout buffer from the source context to the transfer agent',
                           action='store_true')

    # policy config args
    policy_group = parser.add_argument_group('policy configurations')
    policy_group.add_argument('--alg',
                              help='the underlying RL algorithm',
                              choices=Algos,
                              type=Algos,
                              nargs='+',
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
                                nargs='+',
                                default=LEARNING_RATES)
    learning_group.add_argument('--batch_size',
                                help='training mini-batch size',
                                type=int,
                                nargs='+',
                                default=BATCH_SIZES)
    learning_group.add_argument('--gamma',
                                help='discount factor for chosen RL algo',
                                type=float,
                                nargs='+',
                                default=GAMMA)
    learning_group.add_argument('--timesteps',
                                help='max number of timesteps for the trained agent',
                                type=lambda x: int(float(x)),
                                default=TOTAL_TIMESTEPS)

    # algo specific
    learning_group.add_argument('--on_policy_n_steps',
                                help='for on-policy algorithms only! number of steps per experience rollout',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=ON_POLICY_N_STEPS)
    learning_group.add_argument('--on_policy_ent_coef',
                                help='for on-policy algorithms only! entropy loss coefficient',
                                type=lambda x: float(x),
                                nargs='+',
                                default=ON_POLICY_ENT_COEF)
    learning_group.add_argument('--ppo_n_epochs',
                                help='for PPO only! number of training epochs per rollout',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=PPO_N_EPOCHS)
    learning_group.add_argument('--off_policy_learning_starts',
                                help='for off-policy algorithms only! minimal number of steps to take before learning',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=OFF_POLICY_LEARNING_STARTS)
    learning_group.add_argument('--off_policy_train_freq',
                                help='for off-policy algorithms only! number of steps per rollout',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=OFF_POLICY_TRAIN_FREQ)
    learning_group.add_argument('--off_policy_train_freq_episodes',
                                help='for off-policy algorithms only! change `train_freq` to be measured in episodes',
                                action='store_true')
    learning_group.add_argument('--off_policy_gradient_steps',
                                help='for on-policy algorithms only! number of learning steps per rollout. -1 == all',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=OFF_POLICY_GRADIENT_STEPS)
    learning_group.add_argument('--dqn_exploration_timesteps',
                                help='for DQN only! fraction of training using declining epsilon greedy',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=DQN_EXPLORATION_TIMESTEPS)
    learning_group.add_argument('--dqn_target_update_interval',
                                help='for DQN only! number of steps between target network parameter updates',
                                type=lambda x: int(float(x)),
                                nargs='+',
                                default=DQN_TARGET_UPDATE_INTERVAL)

    # RM params
    # TODO make more general
    rm_group = parser.add_argument_group('RM configurations')
    rm_group.add_argument('--goal_state_reward',
                          help='RM reward when reaching the goal state',
                          type=float,
                          nargs='+',
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
                          nargs='+',
                          default=FUEL_RESOLUTIONS)
    rm_group.add_argument('--rs_gamma',
                          help='discount factor for reward shaping',
                          type=float,
                          nargs='+',
                          default=DEFAULT_RS_GAMMA)

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
                             nargs='+')
    model_group.add_argument('--ofe_identity',
                             help='observation features extractor will be the identity function. ignores '
                                  '`--ofe_out_dim` and `--ofe_hidden_dims`',
                             action='store_true')
    model_group.add_argument('--gnn_type',
                             help='the type of GNN to use',
                             default=GNN_TYPE,
                             type=lambda x: getattr(torch_geometric.nn, x),
                             choices=models.GnnWithEdgeAttr.__args__[0].__args__,
                             nargs='+')
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
                             nargs='+')
    model_group.add_argument('--gnn_agg',
                             help='the type of aggregation to perform on RM node embeddings',
                             type=lambda x: getattr(models, x),
                             nargs='+',
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
                           type=lambda x: int(float(x)),
                           default=EVAL_FREQ)
    log_group.add_argument('--max_no_improvement_evals',
                           help='training will stop if no improvement is seen for this many evaluations. does not stop '
                                'if `min_timesteps` has not yet been reached. by default there is no early stopping.',
                           type=lambda x: int(float(x)))
    log_group.add_argument('--min_timesteps',
                           help='minimal number of evaluations that must occur, regardless of the '
                                '--max_no_improvement_evals argument. ignored by default',
                           type=lambda x: int(float(x)))
    log_group.add_argument('--chkp_freq',
                           help='the frequency in which checkpoint models are saved (in steps)',
                           type=lambda x: int(float(x)))

    # technical params
    tech_group = parser.add_argument_group('technical configurations')
    tech_group.add_argument('--num_workers',
                            help='max number of threads for running experiments in parallel',
                            type=int,
                            default=1)
    tech_group.add_argument('--force',
                            help='if set, deletes the existing modela and retrains',
                            action='store_true')

    args = parser.parse_args()

    # handle hidden dim values with 'append' action issue using default value
    if args.mods is None:
        args.mods = MODS
    if args.grid_resolution is None:
        args.grid_resolution = GRID_RESOLUTIONS
    if args.ofe_hidden_dims is None:
        args.ofe_hidden_dims = HIDDEN_DIMS
    if args.gnn_hidden_dims is None:
        args.gnn_hidden_dims = HIDDEN_DIMS

    # evals mutual exclusion
    if args.max_no_improvement_evals is None:
        args.min_timesteps = None
    elif args.min_timesteps is None:
        args.min_timesteps = args.timesteps

    # seeds mutual exclusion
    if args.num_seeds is None:  # --num_seeds not given
        if args.seed is None:  # if seed was given, use only that seed
            args.seed = [SEED[0] * i for i in range(1, NUM_SEEDS + 1)]
    elif args.seed is None:  # num seeds given but no specific seed given.
        args.seed = [SEED[0] * i for i in range(1, args.num_seeds + 1)]
    else:  # both given
        if len(args.seed) == 1:  # can only give 1 seed if num_seeds is given
            args.seed = [args.seed[0] * i for i in range(1, NUM_SEEDS + 1)]
        else:
            raise argparse.ArgumentError(None, 'cannot use `--num_seeds` argument with more than one base `--seed`')
    del args.num_seeds  # in any case remove num seeds from arguments

    # target only defaults
    if args.target_timesteps is None:
        args.target_timesteps = args.timesteps
    if args.target_eval_freq is None:
        args.target_eval_freq = args.eval_freq

    return args


if __name__ == '__main__':
    main()
