import hashlib

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rmrl.utils.misc import sha3_hash
from .experiment import Experiment

import tensorboard as tb

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

import glob


class WithTransferExperiment(Experiment):

    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        # train agent on source and target contexts from scratch (or load if exists)
        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        # create agent for target env with src agent parameters
        transfer_agent = self.new_agent_for_env(tgt_env)
        transfer_agent.set_parameters(src_agent.get_parameters())

        self.train_agent(transfer_agent, tgt_eval_env, f'{sha3_hash(tgt_env.task)}_transfer_from_'
                                                       f'{sha3_hash(src_env.task)}')

        # # reload source agent (resets inner parameters in SB3)
        # src_agent = self.load_agent_for_env(src_env)
        #
        # # transfer everything needed to run src agent on tgt env
        # src_agent.env = tgt_agent.env
        # src_agent.policy.observation_space = tgt_agent.policy.observation_space
        # if isinstance(src_agent, OnPolicyAlgorithm):  # different experience buffers for on/off policy algs
        #     src_agent.rollout_buffer = tgt_agent.rollout_buffer.__class__(
        #         tgt_agent.n_steps,
        #         tgt_agent.observation_space,
        #         tgt_agent.action_space,
        #         device=tgt_agent.device,
        #         gamma=tgt_agent.gamma,
        #         gae_lambda=tgt_agent.gae_lambda,
        #         n_envs=tgt_agent.n_envs
        #     )
        # else:
        #     src_agent.replay_buffer = tgt_agent.replay_buffer.__class__(
        #         tgt_agent.buffer_size,
        #         tgt_agent.observation_space,
        #         tgt_agent.action_space,
        #         device=tgt_agent.device,
        #         optimize_memory_usage=tgt_agent.optimize_memory_usage
        #     )
        #
        # src_agent.set_random_seed(self.cfg.seed)  # reseed for consistency
        # self.train_agent(src_agent, tgt_eval_env, f'{sha3_hash(tgt_env.task)}_transfer_from_'
        #                                           f'{sha3_hash(src_env.task)}')

    def load_tb(self):

        dirnames = glob.glob(str(self.tb_log_dir) + '/*')
        all_dfs = {}

        for dirname in dirnames:

            ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
            dframes = {}
            mnames = ea.Tags()['scalars']

            for n in mnames:
                dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n])
                dframes[n].drop("wall_time", axis=1, inplace=True)
                dframes[n] = dframes[n].set_index("epoch")

            all_dfs[dirname.rsplit('/', 1)[-1]] = pd.concat([v for k, v in dframes.items()], axis=1)

        return all_dfs
