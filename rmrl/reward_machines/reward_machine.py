from abc import ABC, abstractmethod
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from torch_geometric.utils.convert import from_networkx


REWARD_ATTR = 'r'


class RewardMachine(ABC):
    def __init__(self, state_machine: nx.Graph, initial_state_id: int):
        self.G = self.__fill_in_rewards(state_machine)
        self.__orig_rewards_G = self.G.copy()

        assert initial_state_id in self.all_states
        self.__u0 = initial_state_id

    def reshape_rewards(self, potentials: Dict[int, float], gamma):
        for u in self.G:
            for v in self.G[u]:
                self.G[u][v][REWARD_ATTR] += gamma * potentials[v] - potentials[u]

    def reset_rewards(self):
        self.G = self.__orig_rewards_G.copy()

    @property
    @abstractmethod
    def P(self):
        pass

    @property
    def all_states(self):
        return set(self.G.nodes)

    @property
    def U(self) -> set:
        return self.all_states - self.F

    @property
    def u0(self) -> int:
        return self.__u0

    @property
    def F(self) -> set:
        return {u for u in self.G.nodes if not self.G[u]}

    @abstractmethod
    def delta(self, u, props):
        pass

    @abstractmethod
    def L(self, s, a, s_prime):
        pass

    @abstractmethod
    def get_node_feature_attr(self) -> List[Any]:
        pass

    @abstractmethod
    def get_edge_feature_attr(self) -> List[Any]:
        pass

    @property
    def num_propositions(self):
        return len(self.P)

    @property
    def num_states(self):
        return self.G.number_of_nodes()

    def prop_list_to_bitmap(self, props):
        return np.isin(self.P, props).astype(np.float).tolist()

    def draw(self, layout=None, plot_figsize=None, **nxdraw_kwargs):
        if callable(layout):
            pos = layout(self.G)
        else:
            pos = layout

        plt.figure(figsize=plot_figsize)
        nx.draw(self.G, pos, **nxdraw_kwargs)
        plt.show()

    def to_pyg_data(self):
        node_attrs = self.get_node_feature_attr()
        edge_attrs = self.get_edge_feature_attr()
        if REWARD_ATTR not in edge_attrs:  # always include reward attribute
            edge_attrs.append(REWARD_ATTR)

        return from_networkx(self.G, node_attrs, edge_attrs)

    @staticmethod
    def __fill_in_rewards(G):
        G = G.copy()
        for u in G.nodes:
            for v in G[u]:
                if REWARD_ATTR not in G[u][v]:
                    G[u][v][REWARD_ATTR] = 0
        return G

