import numpy as np

from rmrl.reward_machines.reward_machine import *
from gym_subgoal_automata.envs.officeworld.officeworld_env import OfficeWorldEnv


class OfficeWorldRM(RewardMachine):

    def _init(self, goal_vel_reward=1, vel_resolution=0.5):
        pass

    def _delta(self) -> TransitionMap:
        delta = {}

        return delta

    def L(self, s):
        pass
