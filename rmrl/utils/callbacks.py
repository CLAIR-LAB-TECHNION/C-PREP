from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class TrueRewardRMEnvCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        self.__acc_reward = 0
        self.__ep_rewards = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        true_reward = info['no-rm_reward']
        self.__acc_reward += true_reward

        done = self.locals['done'][0]
        if done:
            self.__ep_rewards.append(self.__acc_reward)
            self.__acc_reward = 0

        log_interval = self.locals['log_interval']
        if log_interval is not None and len(self.__ep_rewards) % log_interval == 0:
            self.logger.record("rollout/ep_true_rew_mean",
                               safe_mean(self.__ep_rewards))
            self.__ep_rewards = []
        return True
