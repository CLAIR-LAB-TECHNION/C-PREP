import glob

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from .configurations import SupportedExperiments
from .experiment import Experiment


class WithTransferExperiment(Experiment):
    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        # train agent on source and target contexts from scratch (or load if exists)
        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        self.transfer_agent(src_agent, src_env, tgt_env, tgt_eval_env)

    def transfer_agent(self, src_agent, src_env, tgt_env, tgt_eval_env):
        # get task name
        src_task_name = self.get_env_task_name(src_env)
        tgt_task_name = self.get_env_task_name(tgt_env)
        tsf_task_name = f'{tgt_task_name}_transfer_from_{src_task_name}'

        try:
            tsf_agent = self.load_agent_for_task(tsf_task_name, tgt_env)
        except FileNotFoundError:
            # create agent for target env with src agent parameters
            tsf_agent = self.new_agent_for_env(tgt_env)
            tsf_agent.set_parameters(src_agent.get_parameters())

            tsf_agent = self.train_agent(tsf_agent, tgt_eval_env, tsf_task_name)

        return tsf_agent

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
