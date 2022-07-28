from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.utils import safe_mean

from collections import deque


class TrueRewardRMEnvCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.__acc_reward = 0
        self.__ep_rewards = deque(maxlen=100)  # hard coded maxlen in sb3
        self.__is_off_policy = isinstance(self.model, OffPolicyAlgorithm)

        self.__rollout_count = 0

    def _on_training_start(self) -> None:
        self.log_interval = self.locals['log_interval']

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        true_reward = info['no-rm_reward']
        self.__acc_reward += true_reward

        done = self.locals['dones'][0]
        if done:
            self.__ep_rewards.append(self.__acc_reward)
            self.__acc_reward = 0

            # dump rule for off policy
            if self.__is_off_policy:
                self.__maybe_dump(iterations=len(self.__ep_rewards))  # num iters == num episodes

        return True

    def _on_rollout_end(self):
        self.__rollout_count += 1

        # dump rule for on policy
        if not self.__is_off_policy:
            self.__maybe_dump(self.__rollout_count)  # num iters == num rollouts

    def __maybe_dump(self, iterations):
        if (self.log_interval is not None and          # must have a log interval
                len(self.__ep_rewards) > 0 and         # must complete at least one episode
                iterations % self.log_interval == 0):  # num iters and log interval coincide
            self.logger.record("rollout/ep_true_rew_mean",
                               safe_mean(self.__ep_rewards))
