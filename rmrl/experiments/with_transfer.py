import hashlib

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rmrl.utils.misc import sha3_hash
from .experiment import Experiment

import tensorboard as tb

from .configurations import SupportedExperiments
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

import glob


class WithTransferExperiment(Experiment):

    @property
    def label(self):
        return SupportedExperiments.WITH_TRANSFER

    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        # train agent on source and target contexts from scratch (or load if exists)
        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        # create agent for target env with src agent parameters
        transfer_agent = self.new_agent_for_env(tgt_env)
        transfer_agent.set_parameters(src_agent.get_parameters())


        src_task_name = self.get_env_task_name(src_env)
        tgt_task_name = self.get_env_task_name(tgt_env)
        self.train_agent(transfer_agent, tgt_eval_env, f'{tgt_task_name}_transfer_from_'
                                                       f'{src_task_name}')

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
