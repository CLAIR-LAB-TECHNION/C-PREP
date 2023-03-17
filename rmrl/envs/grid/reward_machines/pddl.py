from ....reward_machines.reward_machine import *

from pddlgym.core import PDDLEnv


class PDDLEnvRM(RewardMachine):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        #TODO delete (this is just for autocomplete)
        self.env: PDDLEnv = self.env

    def _init(self, *args, **kwargs):
        self.domain_file = self.env

    def _delta(self) -> TransitionMap:
        pass

    def L(self, s):
        literals = s.literals

        return s
