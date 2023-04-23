from abc import ABC, abstractmethod
from typing import Dict
from itertools import chain, combinations

import networkx as nx

from .reward_machine import RewardMachine
from ..utils.misc import repr_all_public


class PotentialFunction(ABC):
    @abstractmethod
    def __call__(self, rm: RewardMachine, gamma: float) -> Dict[int, float]:
        pass

    def __repr__(self):
        return repr_all_public(self)


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
        all_deltas = {u: rm.delta[u].items() for u in rm.U}

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


class DistFromGoal(PotentialFunction):
    def __init__(self, goal_states):
        self.goal_states = goal_states

    def __call__(self, rm: RewardMachine, gamma: float) -> Dict[int, float]:
        # container to keep minimum distance to goal for each goal state.
        # initialize for all states with maximum path length + 1, so nodes that cannot reach the goal will have minimum
        # potential
        shortest_path_lengths = {u: rm.num_states for u in rm.all_states}

        # iterate over all goal states and find the shortest path to each
        for goal_state in self.goal_states:
            shortest_path_lengths_to_goal = nx.shortest_path_length(rm.G, target=goal_state)

            # update the shortest paths container with the minimum of current saved distance and new distance
            shortest_path_lengths.update({k: min(l, shortest_path_lengths.get(k, float('inf')))
                                         for k, l in shortest_path_lengths_to_goal.items()})

        # return gamma ** dist_to_nearest_goal as the potential function
        return {k: gamma ** l for k, l in shortest_path_lengths.items()}
