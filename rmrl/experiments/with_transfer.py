import glob

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from .configurations import Algos
from .experiment import Experiment

TRANSFER_FROM_MIDFIX = '_transfer_from_'


class WithTransferExperiment(Experiment):
    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        # set tgt ohe settings
        tgt_env.env.env.ohe_start = self.cfg.num_src_samples
        tgt_env.env.env.set_fixed_contexts(tgt_env.env.env.fixed_contexts)
        tgt_eval_env.env.env.ohe_start = self.cfg.num_src_samples
        tgt_eval_env.env.env.set_fixed_contexts(tgt_eval_env.env.env.fixed_contexts)

        # train agent on source and target contexts from scratch (or load if exists)
        src_agent = self.get_agent_for_env(src_env, src_eval_env)

        self._is_tgt = True
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        self.transfer_agent(src_env, tgt_env, tgt_eval_env)

    def transfer_agent(self, src_env, tgt_env, tgt_eval_env):
        # get task name
        src_task_name = self.get_env_task_name(src_env)
        tgt_task_name = self.get_env_task_name(tgt_env)
        tsf_task_name = f'{tgt_task_name}{TRANSFER_FROM_MIDFIX}{src_task_name}'

        try:
            tsf_agent = self.load_agent_for_task(tsf_task_name, tgt_env)
        except FileNotFoundError:
            # create agent new for target env
            tsf_agent = self.new_agent_for_env(tgt_env)

            # update parameters from src agent
            src_agent = self.load_agent_for_env(src_env,
                                                force_load=True,  # ignore forced retraining
                                                model_name=self.cfg.exp_kwargs['transfer_model'])  # load desired model
            tsf_agent.set_parameters(src_agent.get_parameters())

            if self.cfg.exp_kwargs['keep_timesteps']:
                tsf_agent.num_timesteps = src_agent.num_timesteps

                if self.cfg.alg == Algos.DQN:
                    # to allow exploration fraction to match the given exploration timesteps:
                    # - calc exploration_timesteps
                    # - divide by new max timesteps
                    tsf_agent.exploration_fraction = (
                        (tsf_agent.exploration_fraction * self.cfg.max_timesteps) /
                        (src_agent.num_timesteps + self.cfg.max_timesteps)
                    )

            # train agent
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
                dframes[n] = dframes[n].loc[~dframes[n].index.duplicated(keep='first')]

            all_dfs[dirname.rsplit('/', 1)[-1]] = pd.concat([v for k, v in dframes.items()], axis=1)

        return all_dfs
