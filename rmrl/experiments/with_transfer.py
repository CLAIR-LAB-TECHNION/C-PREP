import hashlib

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rmrl.utils.misc import sha3_hash
from .experiment import Experiment


class WithTransferExperiment(Experiment):

    def _run(self, envs, eval_envs):
        src_env, tgt_env = envs
        src_eval_env, tgt_eval_env = eval_envs

        src_agent = self.get_agent_for_env(src_env, src_eval_env)
        tgt_agent = self.get_agent_for_env(tgt_env, tgt_eval_env)

        # transfer everything needed to run src agent on tgt env
        src_agent.env = tgt_agent.env
        src_agent.policy.observation_space = tgt_agent.policy.observation_space
        if isinstance(src_agent, OnPolicyAlgorithm):  # different experience buffers for on/off policy algs
            src_agent.rollout_buffer = tgt_agent.rollout_buffer.__class__(
                tgt_agent.n_steps,
                tgt_agent.observation_space,
                tgt_agent.action_space,
                device=tgt_agent.device,
                gamma=tgt_agent.gamma,
                gae_lambda=tgt_agent.gae_lambda,
                n_envs=tgt_agent.n_envs
            )
        else:
            src_agent.replay_buffer = tgt_agent.replay_buffer.__class__(
                tgt_agent.buffer_size,
                tgt_agent.observation_space,
                tgt_agent.action_space,
                device=tgt_agent.device,
                optimize_memory_usage=tgt_agent.optimize_memory_usage
            )

        self.train_agent(src_agent, tgt_eval_env, f'{sha3_hash(tgt_env.task)}_transfer_from_'
                                                  f'{sha3_hash(src_env.task)}')
