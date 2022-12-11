from .experiment import Experiment
from .configurations import SupportedExperiments


class NoTransferExperiment(Experiment):
    def _run(self, envs, eval_envs):
        env, tgt_env = envs
        eval_env, tgt_eval_env = eval_envs

        self._tgt_for_test = tgt_eval_env

        self.get_agent_for_env(env, eval_env)
