from functools import reduce
import re

import gym
import numpy as np

from multi_taxi import single_taxi_v0, wrappers
from multi_taxi.world.entities import PASSENGER_NOT_IN_TAXI
from multi_taxi.world.domain_map import DomainMap

from rmrl.context.multitask_env import MultiTaskWrapper
from rmrl.utils.misc import split_pairs


def fixed_entities_env(initial_task=None, change_task_on_reset=False, **env_kwargs):
    env = single_taxi_v0.gym_env(**env_kwargs)
    env = FixedLocsWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset)
    env = NoPassLocDstWrapper(env)

    return env


def changing_map_env(initial_task=None, change_task_on_reset=False, **env_kwargs):
    env = single_taxi_v0.gym_env(**env_kwargs)
    env = ChangeMapWrapper(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset)
    env = FixedLocsAddition(env, initial_task=initial_task, change_task_on_reset=change_task_on_reset)
    return env


class NoPassLocDstWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        meanings = self.env.unwrapped.get_observation_meanings()
        remove_idxs = set()
        for i in range(len(meanings)):
            if re.match(r'passenger_.*_(location|destination)_.*', meanings[i]):
                remove_idxs.add(i)
        all_idxs = set(range(len(self.env.observation_space)))
        self.keep_idxs = sorted(all_idxs - remove_idxs)

    def reset(self, **kwargs):
        obs = super().reset()
        return obs[self.keep_idxs]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs[self.keep_idxs], reward, done, info

    @property
    def observation_space(self):
        space = self.env.observation_space
        nvec = space.nvec
        return gym.spaces.MultiDiscrete(nvec[self.keep_idxs])


class FixedLocsWrapper(MultiTaskWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True):
        # set constant locations and destinations wrappers
        # this will allow us to control taxi locations and passenger locations and destinations on the fly.
        # self.fixed_env = wrappers.FixedTaxiStartLocationsWrapper(env)
        self.fixed_env = wrappers.FixedPassengerStartLocationsWrapper(env)
        self.fixed_env = wrappers.FixedPassengerDestinationsWrapper(self.fixed_env)

        super().__init__(env, initial_task, change_task_on_reset)

    def _sample_task(self, n):
        fixed_locs = []
        for _ in range(n):
            # call private method. BAD PRACTICE
            # taxis = self.unwrapped._MultiTaxiEnv__random_taxis()
            passengers = self.unwrapped._MultiTaxiEnv__random_passengers()

            # concat all locations and all destinations
            # taxi_locs = reduce(lambda locs, t: locs + t.location, taxis, tuple())
            passenger_locs = reduce(lambda locs, p: locs + p.location, passengers, tuple())

            if self.unwrapped.pickup_only:
                passenger_dsts = tuple()
            else:
                passenger_dsts = reduce(lambda dsts, p: dsts + p.destination, passengers, tuple())

            # save locations and destinations
            # fixed_locs.append((taxi_locs, passenger_locs, passenger_dsts))
            fixed_locs.append((passenger_locs, passenger_dsts))

        return fixed_locs

    def _set_task(self, task):
        # override `set_task to update env wrappers' locations
        # new task will only take effect on reset due to uncertainty regarding reset

        # set wrappers for deterministic locations on reset
        self.__set_wrappers(task)

        # set locations in environment
        # we do this to enable changing the environment in the middle of an episode
        # self.__set_locations()

    def _get_hcv_rep(self, task):
        passenger_loc, passenger_dst = task
        return np.array(passenger_loc + passenger_dst)

    def reset(self, **kwargs):
        super().reset(**kwargs)

        # use wrapped env for reset to set the entities in the right place
        # wrapper returns a dictionary for one taxi
        return self.fixed_env.reset(**kwargs)

    def __set_wrappers(self, task):
        # taxi_loc, passenger_loc, passenger_dst = task
        passenger_loc, passenger_dst = task

        # outer wrapper controls destinations
        # set locations in destinations wrapper
        self.fixed_env.locs = split_pairs(passenger_dst)

        # next wrapper controls passenger locations
        # set locations in locations wrapper
        self.fixed_env.env.locs = split_pairs(passenger_loc)

        # next next wrapper controls taxi locations
        # set locations in locations wrapper
        # self.fixed_env.env.env.locs = split_pairs(taxi_loc)

    def __set_locations(self):
        # outer wrapper controls destinations
        # set destinations using dst wrapper methods
        self.fixed_env.set_locations()

        # next wrapper controls passenger locations
        # set locations using loc wrapper method
        self.fixed_env.env.set_locations()

        # next next wrapper controls taxi locations
        # set locations using loc wrapper method
        # self.fixed_env.env.env.set_locations()

        # remove passengers from carrying taxis
        s = self.fixed_env.state()
        for p in s.passengers:
            p.carrying_taxi = PASSENGER_NOT_IN_TAXI
        for t in s.taxis:
            t.passengers = set()
        self.fixed_env.unwrapped.set_state(s)


class FixedLocsAddition(MultiTaskWrapper):
    def __init__(self, env: MultiTaskWrapper, initial_task=None, change_task_on_reset=True):
        # force no changing task on reset for original and location fixer
        env.change_task_on_reset = False
        self.fixed_locs_env = FixedLocsWrapper(env, initial_task, change_task_on_reset=False)

        super().__init__(env, initial_task, change_task_on_reset)

    def _sample_task(self, n):
        orig_task = self.env.sample_task(n)
        locs_task = self.fixed_locs_env.sample_task(1) * n
        return list(zip(locs_task, orig_task))

    def _set_task(self, task):
        locs_task, orig_task = task
        self.env.task = orig_task
        self.fixed_locs_env.task = locs_task

    def _get_hcv_rep(self, task):
        _, orig_task = task
        return self.env._get_hcv_rep(orig_task)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.fixed_locs_env.reset(**kwargs)


class ChangeMapWrapper(MultiTaskWrapper):
    def __init__(self, env, initial_task=None, change_task_on_reset=True):
        self.empty_map = self.__make_empty_map(env.unwrapped.domain_map.domain_map)
        self.wall_locs = [(i, j)
                          for i in range(len(self.empty_map))
                          for j in range(len(self.empty_map[0]))
                          if self.empty_map[i][j] == ':']

        super().__init__(env, initial_task, change_task_on_reset)

    @staticmethod
    def __make_empty_map(domain_map):
        for i in range(1, len(domain_map) - 1):
            for j in range(2, len(domain_map[0]) - 2, 2):
                domain_map[i][j] = ':'

        return domain_map

    def _sample_task(self, n):
        # max number of walls to add
        num_locs = len(self.wall_locs)

        wall_pos_list = []
        for _ in range(n):
            num_locs_to_sample = self._task_np_random.integers(0, num_locs)

            # sample wall positions
            walls_idx = self._task_np_random.choice(range(num_locs), num_locs_to_sample, replace=False)
            walls_pos = [self.wall_locs[idx] for idx in walls_idx]

            # assert walls do not make some areas unreachable
            for j in range(1, len(self.empty_map[0]) - 1):
                # check if entire column is walls (i.e. blocking)
                column_locs = [(i, j) for i in range(1, len(self.empty_map) - 1)]
                if all(loc in walls_pos for loc in column_locs):
                    # column is full of walls
                    # choose one to remove
                    pos_idx_to_remove = self._task_np_random.choice(range(len(column_locs)))
                    wall_loc_to_remove = column_locs[pos_idx_to_remove]
                    walls_pos.remove(wall_loc_to_remove)

            # save sample
            wall_pos_list.append(tuple(walls_pos))

        return wall_pos_list

    def _set_task(self, task):
        map_arr = [[v for v in row] for row in self.empty_map]
        for i, j in task:
            map_arr[i][j] = '|'

        self.unwrapped.domain_map = DomainMap([''.join(row) for row in map_arr])

    def _get_hcv_rep(self, task):
        dm = self.env.unwrapped.domain_map

        # a matrix to indicate wall positions near cells. -1 on width because there is one less wall than cells
        wall_indicator_mat = np.zeros((dm.map_height, dm.map_width - 1))

        # iterate wall positions on map and match to hcv index
        for map_row, map_col in task:
            # map idx row gets -1 for top boundary
            mat_row = map_row - 1

            # map idx col gets -1 for left boundary and divide by 2 to consider cells
            mat_col = (map_col - 1) // 2

            # update indicator matrix
            wall_indicator_mat[mat_row, mat_col] = 1

        # return flat vector
        return wall_indicator_mat.flatten()
