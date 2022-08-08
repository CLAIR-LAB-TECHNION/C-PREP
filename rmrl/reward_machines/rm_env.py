from typing import Callable

import gym
import numpy as np

from rmrl.reward_machines.reward_machine import RewardMachine
from rmrl.context.multitask_env import MultiTaskWrapper

from stable_baselines3.common.custom_spaces import PygData

ORIG_OBS_KEY = 'obs'

CUR_STATE_PROPS_KEY = 'cur_abstract_state'

RM_DATA_KEY = 'rm'


class RMEnvWrapper(gym.Wrapper):
    def __init__(self, env: MultiTaskWrapper, rm_fn: Callable[[MultiTaskWrapper], RewardMachine],
                 rm_observations: bool = True, use_rm_reward: bool = True, abstract_state_props=True,
                 multidiscrete_to_box=True):
        super().__init__(env)

        self.use_rm_reward = use_rm_reward
        self.rm_observations = rm_observations
        self.abstract_state_props = abstract_state_props
        self.multidiscrete_to_box = multidiscrete_to_box

        # set RM data
        self.rm_fn = rm_fn
        self.rm = None
        self.rm_data = None
        self.__first_reset = True

        # current state data will be set upon reset
        self.rm_cur_state = None

        self.__prev_obs = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        if self.env.change_task_on_reset or self.__first_reset:
            self.rm = self.rm_fn(self.env)
            self.rm_data = self.rm.to_pyg_data()

        # save initial abstract state and obs as previous
        self.rm_cur_state = self.rm.L(obs)
        self.__prev_obs = obs

        return self._get_new_obs(obs)

    def step(self, action):
        obs, r, d, info = super().step(action)

        info['no-rm_reward'] = r
        if self.use_rm_reward:  # replacing reward with RM reward
            r = 0

        rm_new_state = self.rm.L(obs)

        # check smooth state transitions
        if tuple(self.rm_cur_state) in self.rm.F:  # terminal state
            raise RuntimeError(f'Bad abstract transition:\n'
                               f'prev obs:  {self.__prev_obs}\n'
                               f'action:    {action}\n'
                               f'new obs:   {obs}\n'
                               f'cur state: {self.rm_cur_state}\n'
                               f'new props: {rm_new_state}')
        if rm_new_state not in self.rm.delta[self.rm_cur_state]:  # no such transition
            raise RuntimeError(f'Bad abstract transition:\n'
                               f'prev obs:  {self.__prev_obs}\n'
                               f'action:    {action}\n'
                               f'new obs:   {obs}\n'
                               f'cur state: {self.rm_cur_state}\n'
                               f'new props: {rm_new_state}')

        # get RM rewards
        rm_r = self.rm.delta[self.rm_cur_state][rm_new_state]

        # use RM reward if specified
        if self.use_rm_reward:
            r += rm_r

        # log rm info
        info[f'rm_reward'] = rm_r
        info[f'rm_new_state'] = rm_new_state

        # save previous abstract state and obs
        self.rm_cur_state = rm_new_state
        self.__prev_obs = obs

        # get observation with graph
        new_obs = self._get_new_obs(obs)

        return new_obs, r, d, info

    @property
    def observation_space(self):
        orig_obs_space = self.env.observation_space
        if isinstance(orig_obs_space, gym.spaces.MultiDiscrete) and self.multidiscrete_to_box:
            orig_obs_space = gym.spaces.Box(low=0, high=orig_obs_space.nvec, shape=(len(orig_obs_space.nvec),))

        if not (self.rm_observations or self.abstract_state_props):
            # no special additions to the observation. return raw observations
            return orig_obs_space
        else:
            # make dict. add raw observations
            spaces_dict = {ORIG_OBS_KEY: orig_obs_space}

            # check if to include the abstract state
            if self.abstract_state_props:
                spaces_dict[CUR_STATE_PROPS_KEY] = gym.spaces.Box(0, 1, (self.rm.num_propositions,))

            # check if to include rm graph data
            if self.rm_observations:
                spaces_dict[RM_DATA_KEY] = PygData(
                    node_features_space=gym.spaces.Box(np.inf, -np.inf, self.rm_data.x.shape[1:]),
                    edge_features_space=gym.spaces.Box(np.inf, -np.inf, self.rm_data.edge_attr.shape[1:])
                )
                # spaces_dict[RM_NODE_FEATURES_OBS_KEY] = gym.spaces.Box(-np.inf, np.inf, self.rm_data.x.shape)
                # spaces_dict[RM_EDGE_INDEX_OBS_KEY] = gym.spaces.Box(-np.inf, np.inf, self.rm_data.edge_index.shape)
                # spaces_dict[RM_EDGE_FEATURES_OBS_KEY] = gym.spaces.Box(-np.inf, np.inf, self.rm_data.edge_attr.shape)

            # return a dictionary space
            return gym.spaces.Dict(spaces_dict)

    def _get_new_obs(self, obs):
        orig_obs_space = self.env.observation_space
        if isinstance(orig_obs_space, gym.spaces.MultiDiscrete) and self.multidiscrete_to_box:
            obs = obs.astype(float)

        if not (self.rm_observations or self.abstract_state_props):
            # no special additions to the observation. return raw observations
            return obs
        else:
            # make dict. add raw observations
            obs_dict = {ORIG_OBS_KEY: obs}

            # check if to include the abstract propositions
            if self.abstract_state_props:
                obs_dict[CUR_STATE_PROPS_KEY] = self.rm_cur_state

            # check if to include rm graph data
            if self.rm_observations:
                obs_dict[RM_DATA_KEY] = self.rm_data
                # obs_dict[RM_NODE_FEATURES_OBS_KEY] = self.rm_data.x.numpy()
                # obs_dict[RM_EDGE_INDEX_OBS_KEY] = self.rm_data.edge_index.numpy()
                # obs_dict[RM_EDGE_FEATURES_OBS_KEY] = self.rm_data.edge_attr.numpy()

            # return a dictionary space
            return obs_dict
