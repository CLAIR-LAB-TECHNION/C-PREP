import gym
import numpy as np

from typing import Callable, List
from rmrl.reward_machines.reward_machine import RewardMachine

from torch_geometric.data import Data
from itertools import product
import torch


ORIG_OBS_KEY = 'obs'
RM_OBS_KEY_FORMAT = 'rm{index}_graph'
RM_CUR_STATE_INDICATOR_KEY_FORMAT = 'rm{index}_cur_state'
RM_CUR_PROPS_VECTOR_KEY_FORMAT = 'rm{index}_cur_props'
RM_NODE_FEATURES_OBS_KEY_FORMAT = 'rm{index}_node_features'
RM_EDGE_INDEX_OBS_KEY_FORMAT = 'rm{index}_edge_index'
RM_EDGE_FEATURES_OBS_KEY_FORMAT = 'rm{index}_edge_features'


class RMEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, rm_fn: Callable[[gym.Env], List[RewardMachine]] = lambda x: [],
                 rm_observations: bool = True, change_rms_on_reset: bool = True, add_orig_reward: bool = False,
                 abstract_state_indicator=False, abstract_props_vector=False, multidiscrete_to_box=True):
        super().__init__(env)

        self.change_rms_on_reset = change_rms_on_reset
        self.add_orig_reward = add_orig_reward
        self.rm_observations = rm_observations
        self.abstract_state_indicator = abstract_state_indicator
        self.abstract_props_vector = abstract_props_vector
        self.multidiscrete_to_box = multidiscrete_to_box
        self.rm_fn = rm_fn
        self.rms = []
        self.rms_data = []
        self.rm_cur_states = []
        self.rm_cur_props = []

        self.__first_time_set_rms_toggle = True
        self.__prev_obs = None

        self.cur_info = {}

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        self.cur_info = {}

        if self.change_rms_on_reset or self.__first_time_set_rms_toggle:
            self._set_rms()
            self.__first_time_set_rms_toggle = False
        self._reset_cur_states(obs)

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
            info[f'rm{i}_new_state'] = new_state
            r += rm_r
            self.rm_cur_states[i] = new_state

        self.__prev_obs = obs
        new_obs = self._get_new_obs(obs)
        self.cur_info = info

        return new_obs, r, d, info

    @property
    def observation_space(self):
        orig_obs_space = self.env.observation_space
        if isinstance(orig_obs_space, gym.spaces.MultiDiscrete) and self.multidiscrete_to_box:
            orig_obs_space = gym.spaces.Box(low=0, high=orig_obs_space.nvec, shape=(len(orig_obs_space.nvec),))

        if not (self.rm_observations or self.abstract_state_indicator or self.abstract_props_vector):
            # no special additions to the observation. return raw observations
            return orig_obs_space
        else:
            # make dict. add raw observations
            spaces_dict = {ORIG_OBS_KEY: orig_obs_space}

            # check if to include the abstract state
            if self.abstract_state_indicator:
                for i, rm in enumerate(self.rms):
                    spaces_dict[self.__cur_state_key(i)] = gym.spaces.Box(0, 1, (rm.num_states,))

            # check if to include the abstract props
            if self.abstract_props_vector:
                for i, rm in enumerate(self.rms):
                    spaces_dict[self.__cur_props_key(i)] = gym.spaces.Box(0, 1, (rm.num_propositions,))

            # check if to include rm graph data
            if self.rm_observations:
                for i, rm_data in enumerate(self.rms_data):
                    spaces_dict[self.__rm_key(i)] = PygData(rm_data.x.shape[1:], rm_data.edge_attr.shape[1:])

            # return a dictionary space
            return gym.spaces.Dict(spaces_dict)

    @staticmethod
    def __rm_key(i):
        return RM_OBS_KEY_FORMAT.format(index=i)

    @staticmethod
    def __cur_state_key(i):
        return RM_CUR_STATE_INDICATOR_KEY_FORMAT.format(index=i)

    @staticmethod
    def __cur_props_key(i):
        return RM_CUR_PROPS_VECTOR_KEY_FORMAT.format(index=i)

    def _get_new_obs(self, obs):
        orig_obs_space = self.env.observation_space
        if isinstance(orig_obs_space, gym.spaces.MultiDiscrete) and self.multidiscrete_to_box:
            obs = obs.astype(float)

        if not (self.rm_observations or self.abstract_state_indicator or self.abstract_props_vector):
            # no special additions to the observation. return raw observations
            return obs
        else:
            # make dict. add raw observations
            obs_dict = {ORIG_OBS_KEY: obs}

            # check if to include the abstract state
            if self.abstract_state_indicator:
                for i, rm in enumerate(self.rms):
                    cur_state = self.rm_cur_states[i]
                    obs_dict[self.__cur_state_key(i)] = rm.state_indicators[cur_state]

            # check if to include the abstract propositions
            if self.abstract_props_vector:
                for i, rm in enumerate(self.rms):
                    obs_dict[self.__cur_props_key(i)] = self.rm_cur_props[i]

            # check if to include rm graph data
            if self.rm_observations:
                for i, rm_data in enumerate(self.rms_data):
                    obs_dict[self.__rm_key(i)] = rm_data

            # return a dictionary space
            return obs_dict

    def _set_rms(self):
        self.rms = self.rm_fn(self.env)
        if self.rm_observations:
            self.rms_data = [rm.to_pyg_data() for rm in self.rms]

    def _reset_cur_states(self, s):
        self.rm_cur_states = [rm.u0(s) for rm in self.rms]
        self.rm_cur_props = [rm.prop_list_to_bitmap([]) for rm in self.rms]  # set to 0 vector of correct length


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
