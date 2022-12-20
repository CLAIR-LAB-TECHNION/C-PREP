from typing import Callable

import gym
import numpy as np
from tqdm.auto import tqdm

from rmrl.reward_machines.reward_machine import RewardMachine
from rmrl.context.multitask_env import MultiTaskWrapper

from rmrl.utils.custom_spaces import PygData

ORIG_OBS_KEY = 'obs'

OHE_CTX_KEY = 'ohe'
HCV_CTX_KEY = 'hcv'

CUR_STATE_PROPS_KEY = 'cur_abstract_state'
NEXT_DESIRED_STATE_PROPS_KEY = 'next_desired_state'

RM_DATA_KEY = 'rm'


class RMEnvWrapper(gym.Wrapper):
    def __init__(self, env: MultiTaskWrapper, rm_fn: Callable[[MultiTaskWrapper], RewardMachine],
                 rm_observations: bool = True, use_rm_reward: bool = True, abstract_state_props=True,
                 next_desired_state_props: bool = False, ohe_ctx: bool = False, hcv_ctx: bool = False,
                 multidiscrete_to_box=True):
        super().__init__(env)

        self.use_rm_reward = use_rm_reward
        self.rm_observations = rm_observations
        self.abstract_state_props = abstract_state_props
        self.next_desired_state_props = next_desired_state_props
        self.ohe_ctx = ohe_ctx
        self.hcv_ctx = hcv_ctx
        self.multidiscrete_to_box = multidiscrete_to_box

        # set RM data
        self.rm_fn = rm_fn
        self.rm = None
        self.rm_data = None
        self.__first_reset = True

        # current state data will be set upon reset
        self.rm_cur_state = None
        self.rm_cur_state_idx = None

        self.__prev_obs = None

        # fixed rms for efficiency
        self.fixed_rms = {}
        self.fixed_rms_data = {}

        # for random seeding
        self._np_random = np.random.default_rng()

    def get_fixed_task_rms(self, tasks):
        out = {}

        multitask_env = self.first_multitask_wrapper

        # stop change task on reset. keep old setting to keep consistent
        old_change_task_on_reset = self.env.change_task_on_reset
        multitask_env.change_task_on_reset = False

        for task in tqdm(tasks, desc='generating rms for context set'):
            multitask_env.task = task
            multitask_env.reset()
            out[task] = self.rm_fn(multitask_env)

        # revert to old task on reset setting
        multitask_env.change_task_on_reset = old_change_task_on_reset

        return out

    @property
    def first_multitask_wrapper(self):
        env = self.env
        while not isinstance(env, MultiTaskWrapper):
            env = env.env

        return env

    def set_fixed_rms(self, fixed_rms, fixed_rms_data):
        self.fixed_rms = fixed_rms
        self.fixed_rms_data = fixed_rms_data

    def seed(self, seed=None):
        super().seed(seed)
        self._np_random = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        if self.env.change_task_on_reset or self.__first_reset:
            self.__first_reset = False

            task = self.env.task
            if task not in self.fixed_rms:
                self.fixed_rms[self.env.task] = self.rm_fn(self.env)
                self.fixed_rms_data[self.env.task] = self.fixed_rms[self.env.task].to_pyg_data()

            self.rm = self.fixed_rms[self.env.task]
            self.rm_data = self.fixed_rms_data[self.env.task]

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

            # check if to include OHE context representation
            if self.ohe_ctx:
                ohe_value = next(iter(self.env.fixed_contexts_ohe_rep.values()))
                spaces_dict[OHE_CTX_KEY] = gym.spaces.Box(0, 1, ohe_value.shape)

            # check if to include HCV context representation
            if self.hcv_ctx:
                hcv_value = next(iter(self.env.fixed_contexts_hcv_rep.values()))
                spaces_dict[HCV_CTX_KEY] = gym.spaces.Box(0, 1, hcv_value.shape)

            # check if to include the current abstract state
            if self.abstract_state_props:
                spaces_dict[CUR_STATE_PROPS_KEY] = gym.spaces.Box(0, 1, (self.rm.num_propositions,))

            # check if to include the next desired abstract state
            if self.next_desired_state_props:
                spaces_dict[NEXT_DESIRED_STATE_PROPS_KEY] = gym.spaces.Box(0, 1, (self.rm.num_propositions,))

            # check if to include rm graph data
            if self.rm_observations:
                spaces_dict[RM_DATA_KEY] = PygData(
                    node_features_space=gym.spaces.Box(np.inf, -np.inf, self.rm_data.x.shape[1:]),
                    edge_features_space=gym.spaces.Box(np.inf, -np.inf, self.rm_data.edge_attr.shape[1:])
                )

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

            # check if to include OHE context representation
            if self.ohe_ctx:
                obs_dict[OHE_CTX_KEY] = self.env.fixed_contexts_ohe_rep[self.env.task]

            # check if to include HCV context representation
            if self.hcv_ctx:
                obs_dict[HCV_CTX_KEY] = self.env.fixed_contexts_hcv_rep[self.env.task]

            # check if to include the abstract propositions
            if self.abstract_state_props:
                obs_dict[CUR_STATE_PROPS_KEY] = np.array(self.rm_cur_state)

            # check if to include the next desired abstract state
            if self.next_desired_state_props:
                # get neighbor edges
                cur_state_atlas = self.rm.G[self.rm_cur_state]

                if cur_state_atlas:  # if edges exist
                    # find max reward in neighbor transitions
                    max_r = max(map(lambda x: x['r'], cur_state_atlas.values()))

                    # find all neighbors with max reward for transition
                    candidates = [s for s, edge_attr in cur_state_atlas.items() if edge_attr['r'] == max_r]
                else:  # no edges
                    candidates = [(0,) * self.rm.num_propositions]  # return all 0

                # randomly choose a candidate
                nds = candidates[self._np_random.choice(range(len(candidates)), 1)[0]]

                obs_dict[NEXT_DESIRED_STATE_PROPS_KEY] = np.array(nds)

            # check if to include rm graph data
            if self.rm_observations:
                obs_dict[RM_DATA_KEY] = self.rm_data
                # obs_dict[RM_NODE_FEATURES_OBS_KEY] = self.rm_data.x.numpy()
                # obs_dict[RM_EDGE_INDEX_OBS_KEY] = self.rm_data.edge_index.numpy()
                # obs_dict[RM_EDGE_FEATURES_OBS_KEY] = self.rm_data.edge_attr.numpy()

            # return a dictionary space
            return obs_dict
