import itertools

import gym
import numpy as np

from rmrl.context.multitask_env import MultiTaskWrapper

MAX_VEL = 10
MIN_VEL = -10
DEFAULT_GOAL_VEL = 5.0

MAX_X = 100
MIN_X = -100

def velocity_env(initial_task=None, change_task_on_reset=False, ohe_classes=None, ohe_start=None, **env_kwargs):
    env = gym.make('HalfCheetah-v4', **env_kwargs)
    env = VelocityWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset,
                          ohe_classes=ohe_classes, ohe_start=ohe_start)
    return env


def velocity_sparse_env(initial_task=None, change_task_on_reset=False, ohe_classes=None, ohe_start=None,
                        goal_tol=0.2, **env_kwargs):
    env = gym.make('HalfCheetah-v4', **env_kwargs)
    env = VelocitySparseWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset,
                                ohe_classes=ohe_classes, ohe_start=ohe_start, goal_tol=goal_tol)
    return env


def location_env(initial_task=None, change_task_on_reset=False, ohe_classes=None, ohe_start=None,
                 goal_tol=0.2, **env_kwargs):
    env = gym.make('HalfCheetah-v4', **env_kwargs, exclude_current_positions_from_observation=False)
    env = LocationWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset,
                          ohe_classes=ohe_classes, ohe_start=ohe_start)
    return env


def lap_runner_env(initial_task=None, change_task_on_reset=False, ohe_classes=None, ohe_start=None, **env_kwargs):
    env = gym.make('HalfCheetah-v4', **env_kwargs, exclude_current_positions_from_observation=False)
    env = LapsRunnerWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset,
                            ohe_classes=ohe_classes, ohe_start=ohe_start)
    return env


class VelocityWrapper(MultiTaskWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None):
        super().__init__(env, initial_task, change_task_on_reset, ohe_classes, ohe_start)
        self.cur_vel = None

    def _set_task(self, task):
        pass  # only use the `task` value as is

    @property
    def goal_vel(self):
        return self.task

    def reset(self, **kwargs):
        self.cur_vel = 0
        # return super().reset(**kwargs)
        obs, info = super().reset(**kwargs)
        return np.concatenate([np.array([self.cur_vel]), obs]), info

    def step(self, action):
        obs, reward, done, t, info = self.env.step(action)

        self.cur_vel = info['x_velocity']

        # set reward sed on distance from goal velocity
        reward_vel = -abs(self.goal_vel - self.cur_vel)
        new_reward = reward_vel + info['reward_ctrl']  # set reward based on control

        # add separate velocity reward to info dict
        info['reward_vel'] = reward_vel

        # return obs, new_reward, done, t, info
        return np.concatenate([np.array([self.cur_vel]), obs]), new_reward, done, t, info

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (super().observation_space.shape[0] + 1,))

    def _sample_task(self, n):
        return self._task_np_random.uniform(low=MIN_VEL, high=MAX_VEL, size=n)

    def _get_hcv_rep(self, task):
        return np.array([task])


class VelocitySparseWrapper(VelocityWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None,
                 goal_tol=0.2):
        super().__init__(env, initial_task, change_task_on_reset, ohe_classes, ohe_start)
        self.goal_tol = goal_tol

    def step(self, action):
        obs, reward, done, t, info = super().step(action)  # sets `cur_vel` to the correct value

        # calculate sparse reward and new reward with control penalty
        sparse_reward = 1 if abs(self.goal_vel - self.cur_vel) <= self.goal_tol else 0
        new_reward = sparse_reward + info['reward_ctrl']

        # add sparse reward to info
        info['reward_sparse'] = sparse_reward

        return obs, new_reward, done, t, info


class LocationWrapper(MultiTaskWrapper):
    goal_list = list(range(1, 32, 2))

    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None):
        super().__init__(env, initial_task, change_task_on_reset, ohe_classes, ohe_start)
        self.cur_vel = None
        self.goal_visited = False

    @property
    def goal_pos(self):
        return self.task

    def step(self, action):
        obs, reward, done, t, info = self.env.step(action)

        self.cur_vel = info['x_velocity']
        cur_pos = info['x_position']

        # check in section status
        pos_idx = self.goal_list.index(self.goal_pos)
        goal_thresh_crossed = cur_pos >= self.goal_pos
        under_next_thresh = pos_idx == len(self.goal_list) - 1 or cur_pos <= self.goal_list[pos_idx + 1]
        in_goal_section = goal_thresh_crossed and under_next_thresh

        if in_goal_section:
            self.goal_visited = True

        if self.goal_visited and not in_goal_section:
            done = True  # if exiting session then terminate episode

        reward_pos = 1000 if in_goal_section else 0  # reward 1000 if in target section, 0 otherwise

        new_reward = reward_pos + info['reward_ctrl']  # set reward based on control

        # add separate pos reward to info dict
        info['reward_pos'] = reward_pos

        return obs, new_reward, done, t, info

    def reset(self, **kwargs):
        self.cur_vel = 0
        self.goal_visited = False
        return super().reset(**kwargs)

    def _get_hcv_rep(self, task):
        return np.array([task])

    def _set_task(self, task):
        pass  # only use the `task` value as is

    def _sample_task(self, n, unique=False):
        return self._task_np_random.choice(self.goal_list, size=n, replace=not unique)


class LapsRunnerWrapper(MultiTaskWrapper):
    FINISH_LAP_REWARD = 1_000

    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None):
        self.section_thresholds = [1, 3, 5, 7, 9, 11]

        super().__init__(env, initial_task, change_task_on_reset, ohe_classes, ohe_start)

        self.base_checkpoint = self.section_thresholds[0]
        self.target_visited = False

    @property
    def target_checkpoint(self):
        return self.task

    def step(self, action):
        obs, reward, done, t, info = self.env.step(action)
        cur_pos = info['x_position']

        if not self.target_visited:  # going to target
            reward_pos = 0
            if cur_pos >= self.target_checkpoint:
                self.target_visited = True
        else:  # returning from target
            if cur_pos <= self.base_checkpoint:  # back to beginning. get reward
                self.target_visited = False  # reset visited status for task continue
                reward_pos = self.FINISH_LAP_REWARD
            else:  # still going back to 0. no reward
                reward_pos = 0

        new_reward = reward_pos + info['reward_ctrl']  # set reward based on control

        # add separate pos reward to info dict
        info['reward_lap'] = reward_pos

        # return obs, new_reward, done, t, info
        return np.concatenate([obs, np.array([self.target_visited])]), new_reward, done, t, info

    def reset(self, **kwargs):
        self.target_visited = False
        obs, info = super().reset(**kwargs)
        return np.concatenate([obs, np.array([self.target_visited])]), info

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (super().observation_space.shape[0] + 1,))

    def _get_hcv_rep(self, task):
        return np.array([task])

    def _set_task(self, task):
        self.target_visited = False
        self._goal_pos_idx = self.section_thresholds.index(self.target_checkpoint)

    def _sample_task(self, n, unique=False):
        return self._task_np_random.choice(self.section_thresholds, size=n, replace=not unique)
