import gym
import numpy as np

from typing import Callable, List
from rmrl.reward_machines.reward_machine import RewardMachine

from torch_geometric.data import Data
from itertools import product
import torch


ORIG_OBS_KEY = 'obs'
RM_OBS_KEY_FORMAT = 'rm{index}'
RM_NODE_FEATURES_OBS_KEY_FORMAT = 'rm{index}_node_features'
RM_EDGE_INDEX_OBS_KEY_FORMAT = 'rm{index}_edge_index'
RM_EDGE_FEATURES_OBS_KEY_FORMAT = 'rm{index}_edge_features'


class RMEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, rm_fn: Callable[[gym.Env], List[RewardMachine]] = lambda x: [],
                 rm_observations: bool = True, change_rms_on_reset: bool = True, add_orig_reward: bool = False):
        super().__init__(env)

        self.change_rms_on_reset = change_rms_on_reset
        self.add_orig_reward = add_orig_reward
        self.rm_observations = rm_observations
        self.rm_fn = rm_fn
        self.rms = []
        self.rms_data = []
        self.rm_cur_states = []

        self._set_rms()
        self._reset_cur_states()
        self.__first_time_set_rms_toggle = True
        self.__prev_obs = None

        self.cur_info = {}

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        self.cur_info = {}

        if self.change_rms_on_reset or self.__first_time_set_rms_toggle:
            self._set_rms()
            self.__first_time_set_rms_toggle = False
        self._reset_cur_states()

        self.__prev_obs = obs
        return self._get_new_obs(obs)

    def step(self, action):
        obs, r, d, info = super().step(action)

        info['no-rm_reward'] = r
        if not self.add_orig_reward:
            r = 0

        for i in range(len(self.rms)):
            rm, cur_state = self.rms[i], self.rm_cur_states[i]
            propositions = rm.L(self.__prev_obs, action, obs)
            new_state, rm_r = rm.delta(cur_state, propositions)
            info[f'rm{i}_reward'] = rm_r
            info[f'rm{i}_propositions'] = propositions
            r += rm_r

        self.__prev_obs = obs
        new_obs = self._get_new_obs(obs)
        self.cur_info = info

        return new_obs, r, d, info

    # @property
    # def observation_space(self):
    #     if self.rm_observations:
    #         spaces_dict = {ORIG_OBS_KEY: self.env.observation_space}
    #
    #         for i, rm_data in enumerate(self.rms_data):
    #             spaces_dict[self.__rm_key(i)] = PygData(rm_data.x.shape[1:], rm_data.edge_attr.shape[1:])
    #
    #         return gym.spaces.Dict(spaces_dict)
    #     else:
    #         return self.env.observation_space
    #
    # @staticmethod
    # def __rm_key(i):
    #     return RM_OBS_KEY_FORMAT.format(index=i)

    @property
    def observation_space(self):
        if self.rm_observations:
            spaces_dict = {ORIG_OBS_KEY: self.env.observation_space}

            for i, rm in enumerate(self.rms):
                rm_data = rm.to_pyg_data()
                spaces_dict[self.__nf_key(i)] = gym.spaces.Box(-np.inf, np.inf, rm_data.x.shape)
                spaces_dict[self.__ei_key(i)] = gym.spaces.Box(-np.inf, np.inf, rm_data.edge_index.shape)
                spaces_dict[self.__ef_key(i)] = gym.spaces.Box(-np.inf, np.inf, rm_data.edge_attr.shape)

            return gym.spaces.Dict(spaces_dict)
        else:
            return self.env.observation_space

    def _get_new_obs(self, obs):
        if self.rm_observations:
            new_obs = {ORIG_OBS_KEY: obs}

            for i, graph_data in enumerate(self.rms_data):
                new_obs[self.__nf_key(i)] = graph_data.x.numpy()
                new_obs[self.__ei_key(i)] = graph_data.edge_index.numpy()
                new_obs[self.__ef_key(i)] = graph_data.edge_attr.numpy()

            return new_obs
        else:
            return obs

    @staticmethod
    def __nf_key(i):
        return RM_NODE_FEATURES_OBS_KEY_FORMAT.format(index=i)

    @staticmethod
    def __ei_key(i):
        return RM_EDGE_INDEX_OBS_KEY_FORMAT.format(index=i)

    @staticmethod
    def __ef_key(i):
        return RM_EDGE_FEATURES_OBS_KEY_FORMAT.format(index=i)

    def _set_rms(self):
        self.rms = self.rm_fn(self.env)
        self.rms_data = [rm.to_pyg_data() for rm in self.rms]

    def _reset_cur_states(self):
        self.rm_cur_states = [rm.u0 for rm in self.rms]


class PygData(gym.spaces.Space):
    def __init__(self, nf_shape, ef_shape, max_nodes=100, dtype=np.float32, seed=None):
        super().__init__(nf_shape + ef_shape, dtype, seed)

        self.nf_box = gym.spaces.Box(-np.inf, np.inf, shape=nf_shape, dtype=dtype, seed=seed)
        self.ef_box = gym.spaces.Box(-np.inf, np.inf, shape=ef_shape, dtype=dtype, seed=seed)
        self.max_nodes = max_nodes

    def sample(self):
        num_nodes = self.np_random.randint(1, self.max_nodes)
        num_edges = self.np_random.randint(1, num_nodes ** 2)

        nf = torch.tensor([self.nf_box.sample() for _ in range(num_nodes)])
        ef = torch.tensor([self.ef_box.sample() for _ in range(num_edges)])

        possible_edges = np.array(list(product(range(num_nodes), range(num_nodes))))
        edge_idx = torch.from_numpy(possible_edges[
            self.np_random.choice(len(possible_edges), size=num_edges, replace=False)
        ])

        return Data(nf, edge_idx, ef)

    def contains(self, x: Data):
        if not isinstance(x, Data):
            return False

        if x.x.shape[0] > self.max_nodes:
            return False

        if not x.x.shape[1:] != self.nf_box.shape:
            return False

        if x.edge_attr is None and self.ef_box.shape != (0,):
            return False
        elif x.edge_attr.shape[1:] != self.ef_box.shape:
            return False

        return True

    def __repr__(self):
        return f'PygData(nf_box={self.nf_box}, ef_box={self.ef_box})'
