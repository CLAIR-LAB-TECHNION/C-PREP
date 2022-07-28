from .experiment import Experiment


class WithTransferExperiment(Experiment):

    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        src_agent.env = tgt_agent.env
        self.train_agent(src_agent, tgt_eval_env,  f'{tgt_env.task}_transfer_from_{src_env.task}')
