from abc import ABC

import numpy as np
import gym_subgoal_automata
from gym_subgoal_automata.envs.officeworld.officeworld_env import OfficeWorldEnv
import gym
from gym.envs.registration import registry

from ...context.multitask_env import MultiTaskWrapper

ENV_PREFIXES = [
    'OfficeWorld',
    'WaterWorld',
    'CraftWorld'
]


class SubgoalAutomataWrapper(MultiTaskWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True, ohe_classes=None, ohe_start=None):
        self.env_name = self._get_env_prefix(env)
        self.all_tasks = self._get_all_tasks()
        super().__init__(env, initial_task, change_task_on_reset, ohe_classes, ohe_start)

    @property
    def num_tasks(self):
        return len(self.all_tasks)

    def _get_env_prefix(self, env):
        env_class = env.unwrapped.__class__.__name__
        for prefix in ENV_PREFIXES:
            if env_class.startswith(prefix):
                return prefix

        raise ValueError(f'bad env class {env_class}')

    def _get_all_tasks(self):
        all_tasks = list(filter(lambda k: k.startswith(self.env_name), registry.keys()))

        # waterworld allows waterworld with custom tasks. ignore this option
        if 'WaterWorld-v0' in all_tasks:
            all_tasks.remove('WaterWorld-v0')

        return all_tasks

    def _set_task(self, task):

        # officeworld requires the `params` argument with map generation instructions
        # use the same maps as in the "using reward machines" paper.
        params = {}
        if task.startswith('OfficeWorld'):
            params[OfficeWorldEnv.MAP_GENERATION_FIELD] = OfficeWorldEnv.PAPER_MAP_GENERATION

        self.env = gym.make(task, params=params)

    def _sample_task(self, n):
        if n > self.num_tasks:
            raise ValueError(f'requested {n} tasks, but only {self.num_tasks} exist')
        else:
            return self.np_random.choice(self.all_tasks, n, replace=False)

    def _get_hcv_rep(self, task):
        return np.array([])
