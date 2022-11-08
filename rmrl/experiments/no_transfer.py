from .experiment import Experiment


class NoTransferExperiment(Experiment):
    def _run(self, envs, eval_envs):
        [env] = envs
        [eval_env] = eval_envs

        self.get_agent_for_env(env, eval_env)
