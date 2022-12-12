from .experiment import Experiment
from .configurations import SupportedExperiments


class NoTransferExperiment(Experiment):
    def _run(self, envs, eval_envs):
        env, tgt_env = envs
        eval_env, tgt_eval_env = eval_envs

        # set tgt ohe settings
        tgt_eval_env.env.env.ohe_start = self.cfg.num_src_samples
        tgt_eval_env.env.env.set_fixed_contexts(tgt_eval_env.env.env.fixed_contexts)

        self._tgt_for_test = tgt_eval_env

        self.get_agent_for_env(env, eval_env)
