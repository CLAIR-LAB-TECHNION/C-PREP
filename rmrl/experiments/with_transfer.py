from .experiment import Experiment
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class WithTransferExperiment(Experiment):

    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        # transfer everything needed to run src agent on tgt env
        src_agent.env = tgt_agent.env
        src_agent.policy.observation_space = tgt_agent.policy.observation_space
        if isinstance(src_agent, OnPolicyAlgorithm):
            src_agent.rollout_buffer = tgt_agent.rollout_buffer
        else:
            src_agent.replay_buffer = tgt_agent.replay_buffer

        self.train_agent(src_agent, tgt_eval_env,  f'{tgt_env.task}_transfer_from_{src_env.task}')
