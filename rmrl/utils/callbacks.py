import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import sync_envs_normalization
from tqdm.auto import tqdm


class LogMeanEpisodeInfoCallback(BaseCallback):
    def __init__(self, info_key, episode_agg=np.mean, verbose: int = 0):
        super().__init__(verbose)
        self.info_key = info_key
        self.episode_agg = episode_agg

        # key of log item
        # e.g. the info "inner_reward" aggregated with sum will be:
        #     rollout/mean_ep_sum_inner_reward
        self.__log_key = f"rollout/mean_ep_{self.episode_agg.__name__}_{self.info_key}"

    def _init_callback(self) -> None:
        self.__ep_infos = [[] for _ in range(self.model.n_envs)  ]
        self.__agg_ep_infos = deque(maxlen=100)
        self.__is_off_policy = isinstance(self.model, OffPolicyAlgorithm)

        self.__rollout_count = 0

    def _on_training_start(self) -> None:
        self.log_interval = self.locals['log_interval']

    def _on_step(self) -> bool:
        for i in range(self.model.n_envs):
            info = self.locals['infos'][i]
            self.__ep_infos[i].append(info[self.info_key] if self.info_key in info else 0)  # self.locals['rewards'][i]

            if self.locals['dones'][i]:
                self.__agg_ep_infos.append(self.episode_agg(self.__ep_infos[i]))
                self.__ep_infos[i] = []

                # dump rule for off policy
                if self.__is_off_policy:
                    self.__maybe_dump(iterations=len(self.__agg_ep_infos))  # num iters == num episodes

        return True

    def _on_rollout_end(self):
        self.__rollout_count += 1

        # dump rule for on policy
        if not self.__is_off_policy:
            self.__maybe_dump(self.__rollout_count)  # num iters == num rollouts

    def __maybe_dump(self, iterations):
        if (self.log_interval is not None and  # must have a log interval
                len(self.__agg_ep_infos) > 0 and  # must complete at least one episode
                iterations % self.log_interval == 0):  # num iters and log interval coincide
            self.logger.record(self.__log_key, safe_mean(self.__agg_ep_infos))


class RMEnvRewardCallback(BaseCallback):
    def _init_callback(self) -> None:
        self.__acc_rewards = [0] * self.model.n_envs
        self.__ep_rewards = deque(maxlen=100)  # hard coded maxlen in sb3
        self.__is_off_policy = isinstance(self.model, OffPolicyAlgorithm)

        self.__rollout_count = 0

    def _on_training_start(self) -> None:
        self.log_interval = self.locals['log_interval']

    def _on_step(self) -> bool:
        for i in range(self.model.n_envs):
            info = self.locals['infos'][0]
            self.__acc_rewards[i] += info['rm_reward'] if 'rm_reward' in info else 0  # self.locals['rewards'][i]

            if self.locals['dones'][i]:
                self.__ep_rewards.append(self.__acc_rewards[i])
                self.__acc_rewards[i] = 0

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
        if (self.log_interval is not None and  # must have a log interval
                len(self.__ep_rewards) > 0 and  # must complete at least one episode
                iterations % self.log_interval == 0):  # num iters and log interval coincide
            self.logger.record("rollout/ep_rm_rew_mean",
                               safe_mean(self.__ep_rewards))


class ProgressBarCallback(BaseCallback):
    def _init_callback(self) -> None:
        self.pbar = tqdm(total=self.model._total_timesteps - self.model._num_timesteps_at_start, desc='training')

    def _on_step(self) -> bool:
        self.pbar.update(self.model.num_timesteps - self.model._num_timesteps_at_start - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


class CustomEvalCallback(EvalCallback):
    def __init__(self,
                 eval_env,
                 callback_on_new_best=None,
                 callback_after_eval=None,
                 n_eval_episodes=5,
                 eval_freq=10000,
                 log_path=None,
                 best_model_save_path=None,
                 deterministic=True,
                 render=False,
                 verbose=1,
                 warn=True,
                 save_buffer=False,
                 logger_prefix=''):

        super().__init__(eval_env,
                         callback_on_new_best,
                         callback_after_eval,
                         n_eval_episodes,
                         eval_freq,
                         log_path,
                         best_model_save_path,
                         deterministic,
                         render,
                         verbose,
                         warn)

        self.save_buffer = save_buffer
        self.logger_prefix = logger_prefix

    def _init_callback(self):
        super()._init_callback()

        # needed for calculating and storing returns
        self._returns_buffer = []
        self._gamma = self.model.gamma  # exists in on-policy and off-policy algorithms
        self._reset_discounts()
        self.evaluations_returns = []

        # set seed for eval env
        self.eval_env.seed(self.model.seed)

    def _reset_discounts(self):
        self._discount = 1
        self._return = 0

    def _log_returns(self, locals_, globals_):
        info = locals_['info']
        r = info['no-rm_reward'] if 'no-rm_reward' in info else locals_['reward']
        self._return += r * self._discount
        self._discount *= self._gamma

        if locals_["done"]:
            self._returns_buffer.append(self._return)
            self._reset_discounts()

    def _on_training_start(self) -> None:
        self.num_timesteps = self.model.num_timesteps
        self._on_step()  # do step on training start for zero-shot testing

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset returns buffer
            self._returns_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_returns,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps - self.model._num_timesteps_at_start)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save returns log if present
                if len(self._returns_buffer) > 0:
                    self.evaluations_returns.append(self._returns_buffer)
                    kwargs = dict(returns=self.evaluations_returns)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.logger_prefix + '_' if self.logger_prefix else ''}eval/mean_reward",
                               float(mean_reward))
            self.logger.record(f"{self.logger_prefix + '_' if self.logger_prefix else ''}eval/mean_ep_length",
                               mean_ep_length)

            if len(self._returns_buffer) > 0:
                mean_return, std_return = np.mean(self._returns_buffer), np.std(self._returns_buffer)
                self._returns_buffer = []
                if self.verbose > 0:
                    print(f"Episode return: {mean_return:.2f} +/- {std_return:.2f}")
                self.logger.record(f"{self.logger_prefix + '_' if self.logger_prefix else ''}eval/mean_return",
                                   mean_return)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/total_timesteps",
                               self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    if self.save_buffer and isinstance(self.model, OffPolicyAlgorithm):
                        self.model.save_replay_buffer(os.path.join(self.best_model_save_path, "best_buffer"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
