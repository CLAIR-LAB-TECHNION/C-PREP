from abc import ABC, abstractmethod
from typing import Dict
from itertools import chain, combinations


from .reward_machine import RewardMachine


class PotentialFunction(ABC):
    @abstractmethod
    def __call__(self, rm: RewardMachine, gamma: float) -> Dict[int, float]:
        pass


class ValueIteration(PotentialFunction):
    def __init__(self, neg_results: bool = False):
        self.neg_results = neg_results

    def __call__(self, rm: RewardMachine, gamma: float, tolerance: float = 1e-7) -> Dict[int, float]:
        v_star = self._value_iteration(rm, gamma, tolerance)
        if self.neg_results:
            return self.__neg_dict(v_star)  # in the paper they did negative. doesn't work correctly for me.
        else:
            return v_star

    @classmethod
    def _value_iteration(cls, rm: RewardMachine, gamma: float, tolerance: float = 1e-7) -> Dict[int, float]:
        v = {u: 0 for u in rm.all_states}
        all_deltas = {u: [rm.delta(u, props) for props in cls.__powerset(rm.P)] for u in rm.U}

        e = 1
        while e > tolerance:
            e = 0
            v_copy = v.copy()  # use the value of the previous iteration when calculating
            for u in rm.U:
                v_prime = max([delta_r + gamma * v_copy[delta_u] for delta_u, delta_r in all_deltas[u]])
                e = max(e, abs(v[u] - v_prime))
                v[u] = v_prime

        return v

    @staticmethod
    def __neg_dict(d):
        return {k: -v for k, v in d.items()}

    @staticmethod
    def __powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
