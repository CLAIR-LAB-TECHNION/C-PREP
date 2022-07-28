from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

from ..context.multitask_env import MultiTaskWrapper

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from torch_geometric.utils.convert import from_networkx


STATE_INDICATOR_ATTR = 'i'
PROPS_VECTOR_ATTR = 'p'
REWARD_ATTR = 'r'

PropsVector = Tuple[bool, ...]
TransitionMap = Dict[PropsVector, Dict[PropsVector, float]]


class RewardMachine(ABC):
    def __init__(self, env, *args, **kwargs):
        self.env = env

        self._init(*args, **kwargs)
        self.delta = self._delta()

        self.G = self._create_state_machine()
        self.__fill_in_rewards()

        # save state machine with original rewards for option to reset reshaping
        self.__orig_rewards_G = self.G.copy()

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    @abstractmethod
    def _delta(self) -> TransitionMap:
        pass

    @abstractmethod
    def L(self, s):
        pass

    def _create_state_machine(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for u in self.delta:
            for v, r in self.delta[u].items():
                G.add_node(u, **{PROPS_VECTOR_ATTR: u})
                G.add_node(v, **{PROPS_VECTOR_ATTR: v})
                G.add_edge(u, v, **{REWARD_ATTR: r})

        return G

    @property
    def num_propositions(self):
        _, some_node_data = next(iter(self.G.nodes(data=True)))
        return len(some_node_data[PROPS_VECTOR_ATTR])

    @property
    def all_states(self):
        return set(self.G.nodes)

    @property
    def U(self):
        return set(self.delta.keys())

    @property
    def F(self):
        return self.all_states - self.U

    @property
    def num_states(self):
        return self.G.number_of_nodes()

    def reshape_rewards(self, potentials: Dict[int, float], gamma):
        for u in self.G:
            for v in self.G[u]:
                new_r = gamma * potentials[v] - potentials[u]
                self.G[u][v][REWARD_ATTR] += new_r
                self.delta[u][v] += new_r

    def draw(self, layout=None, plot_figsize=None, **nxdraw_kwargs):
        if callable(layout):
            pos = layout(self.G)
        else:
            pos = layout

        plt.figure(figsize=plot_figsize)
        nx.draw(self.G, pos, **nxdraw_kwargs)
        plt.show()

    def to_pyg_data(self):
        return from_networkx(G=self.G, group_node_attrs=all, group_edge_attrs=all)

    def __fill_in_rewards(self):
        for u in self.G.nodes:
            for v in self.G[u]:
                if REWARD_ATTR not in self.G[u][v]:
                    self.G[u][v][REWARD_ATTR] = 0


# class RewardMachine(ABC):
#     def __init__(self, *args, use_node_indicator_attr=False, **kwargs):
#         self._init(*args, **kwargs)
#         self.G = self.create_state_machine()
#         self.__fill_in_rewards()
#
#         self.state_indicators = np.eye(self.num_states)
#         if use_node_indicator_attr:
#             self.__add_state_indicator()
#
#         # save state machine with original rewards for option to reset reshaping
#         self.__orig_rewards_G = self.G.copy()
#
#     def reshape_rewards(self, potentials: Dict[int, float], gamma):
#         for u in self.G:
#             for v in self.G[u]:
#                 self.G[u][v][REWARD_ATTR] += gamma * potentials[v] - potentials[u]
#
#     def reset_sm(self):
#         self.G = self.__orig_rewards_G.copy()
#
#     @abstractmethod
#     def _init(self, *args, **kwargs):
#         pass
#
#     @property
#     @abstractmethod
#     def P(self):
#         pass
#
#     @property
#     def all_states(self):
#         return set(self.G.nodes)
#
#     @property
#     def U(self) -> set:
#         return self.all_states - self.F
#
#     @abstractmethod
#     def u0(self, s) -> int:
#         pass
#
#     @property
#     def F(self) -> set:
#         return {u for u in self.G.nodes if not self.G[u]}
#
#     @abstractmethod
#     def L(self, s):
#         pass
#
#     @abstractmethod
#     def delta(self, u, props):
#         pass
#
#     @abstractmethod
#     def new_task(self, task):
#         pass
#
#     @abstractmethod
#     def create_state_machine(self):
#         pass
#
#     def get_node_feature_attr(self) -> List[Any]:
#         return all
#
#     def get_edge_feature_attr(self) -> List[Any]:
#         return all
#
#     @property
#     def num_propositions(self):
#         return len(self.P)
#
#     @property
#     def num_states(self):
#         return self.G.number_of_nodes()
#
#     def prop_list_to_bitmap(self, props):
#         return np.isin(self.P, props).astype(float).tolist()
#
#     def draw(self, layout=None, plot_figsize=None, **nxdraw_kwargs):
#         if callable(layout):
#             pos = layout(self.G)
#         else:
#             pos = layout
#
#         plt.figure(figsize=plot_figsize)
#         nx.draw(self.G, pos, **nxdraw_kwargs)
#         plt.show()
#
#     def to_pyg_data(self):
#         node_attrs = self.get_node_feature_attr()
#         edge_attrs = self.get_edge_feature_attr()
#         if edge_attrs != all and REWARD_ATTR not in edge_attrs:  # always include reward attribute
#             edge_attrs.append(REWARD_ATTR)
#
#         return from_networkx(self.G, node_attrs, edge_attrs)
#
#     def __fill_in_rewards(self):
#         for u in self.G.nodes:
#             for v in self.G[u]:
#                 if REWARD_ATTR not in self.G[u][v]:
#                     self.G[u][v][REWARD_ATTR] = 0
#
#     def __add_state_indicator(self):
#         for u, u_data in self.G.nodes(data=True):
#             u_data[STATE_INDICATOR_ATTR] = self.state_indicators[u]
