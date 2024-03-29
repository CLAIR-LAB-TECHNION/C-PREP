from abc import ABC, abstractmethod

import gym
import numpy as np


class MultiTaskWrapper(gym.Wrapper, ABC):
    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None):
        super().__init__(env)
        self._task_np_random = np.random.default_rng()
        self.fixed_contexts = None
        self.fixed_contexts_ohe_rep = None
        self.fixed_contexts_hcv_rep = None
        self.ohe_classes = ohe_classes
        self.ohe_start = ohe_start

        self.task = initial_task if initial_task is not None else self.sample_task(1)[0]
        self.change_task_on_reset = change_task_on_reset

    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, value):
        self.__task = value
        self._set_task(value)

    def reset(self, **kwargs):
        if self.change_task_on_reset:
            new_task = self.sample_task(1)[0]
            self.task = new_task

        return super().reset(**kwargs)

    def seed(self, seed=None):
        super().seed(seed)
        self._task_np_random = np.random.default_rng(seed)

    def set_fixed_contexts(self, contexts):
        self.fixed_contexts = contexts

        if self.ohe_classes is None:
            # number of ohe classes not given. use given fixed contexts
            self.ohe_classes = len(self.fixed_contexts)

        ohe_vecs = np.eye(self.ohe_classes)
        if self.ohe_start is not None:
            # ohe value starts at the given starting point
            ohe_vecs = ohe_vecs[self.ohe_start:]

        # calculate ohe vectors
        # expected number of contexts to be less that the number of encodings left after `ohe_start`
        self.fixed_contexts_ohe_rep = {t: ohe_vecs[i] for i, t in enumerate(self.fixed_contexts)}

        hcv_vecs = list(map(self._get_hcv_rep, self.fixed_contexts))
        self.fixed_contexts_hcv_rep = {t: v for t, v in zip(self.fixed_contexts, hcv_vecs)}

    @abstractmethod
    def _get_hcv_rep(self, task):
        pass

    @abstractmethod
    def _set_task(self, task):
        pass

    def sample_task(self, n):
        if self.fixed_contexts is None:
            return self._sample_task(n)
        else:
            samples_idx = self._task_np_random.choice(range(len(self.fixed_contexts)), n)
            return [self.fixed_contexts[i] for i in samples_idx]

    @abstractmethod
    def _sample_task(self, n):
        pass


