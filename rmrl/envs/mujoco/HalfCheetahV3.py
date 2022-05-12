import gym
from ..multitask_env import MultiTaskWrapper

MAX_VEL = 10
MIN_VEL = -10

DEFAULT_GOAL_VEL = 1


def velocity_env(initial_goal_vel=None, change_task_on_reset=True):
    env = gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False)
    env = VelocityWrapper(env, initial_task=initial_goal_vel, change_task_on_reset=change_task_on_reset)

    return env


class VelocityWrapper(MultiTaskWrapper):

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # set reward sed on distance from goal velocity
        reward_vel = -abs(self.task - info['x_velocity'])
        new_reward = reward_vel + info['reward_ctrl']  # set reward based on control

        # add separate velocity reward to info dict
        info['reward_vel'] = reward_vel

        return obs, new_reward, done, info

    def sample_task(self, n):
        return self._task_np_random.uniform(low=MIN_VEL, high=MAX_VEL, size=n)
