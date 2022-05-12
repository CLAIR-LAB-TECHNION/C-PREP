from abc import ABC, abstractmethod

import gym
import numpy as np


class MultiTaskWrapper(gym.Wrapper, ABC):
    def __init__(self, env, initial_task=None, change_task_on_reset=True):
        super().__init__(env)

        self.task = initial_task if initial_task is not None else self.sample_task(1)[0]
        self.change_task_on_reset = change_task_on_reset

        self._task_np_random = np.random

    def reset(self, **kwargs):
        if self.change_task_on_reset:
            self.task = self.sample_task(1)[0]

        return super().reset(**kwargs)

    def seed(self, seed=None):
        super().seed()
        self._task_np_random = np.random.default_rng(seed)

    def set_task(self, task):
        self.task = task

    @abstractmethod
    def sample_task(self, n):
        pass
