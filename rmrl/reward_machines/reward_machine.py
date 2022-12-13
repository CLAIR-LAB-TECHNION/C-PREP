from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import from_networkx

STATE_INDICATOR_ATTR = 'i'
PROPS_VECTOR_ATTR = 'p'
REWARD_ATTR = 'r'

PropsVector = Tuple[bool, ...]
TransitionMap = Dict[PropsVector, Dict[PropsVector, float]]


class RewardMachine(ABC):
    def __init__(self, env, *args, rs_gamma=None, **kwargs):
        self.env = env
        self.rs_gamma = rs_gamma

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

    @cached_property
    def num_propositions(self):
        _, some_node_data = next(iter(self.G.nodes(data=True)))
        return len(some_node_data[PROPS_VECTOR_ATTR])

    @cached_property
    def all_states(self):
        return set(self.G.nodes)

    @cached_property
    def U(self):
        return set(self.delta.keys())

    @cached_property
    def F(self):
        return self.all_states - self.U

    @cached_property
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
        data = from_networkx(G=self.G, group_node_attrs=all, group_edge_attrs=all)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        return data

    def __fill_in_rewards(self):
        for u in self.G.nodes:
            for v in self.G[u]:
                if REWARD_ATTR not in self.G[u][v]:
                    self.G[u][v][REWARD_ATTR] = 0

    # method for pickle
    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove unpicklable env.
        state['env'] = None

        return state


class RMFromNX(RewardMachine):

    def _init(self, G: nx.DiGraph, props_vector_attr=None, reward_attr=None):
        if props_vector_attr is not None:
            G = nx.relabel_nodes(G, {n: tuple(G.nodes[n][props_vector_attr]) for n in G.nodes})

        if reward_attr is not None:
            for _, _, edge_data in G.edges(data=True):
                edge_data[REWARD_ATTR] = edge_data.pop(reward_attr)

        self.G = G

    def _delta(self) -> TransitionMap:
        return {u: {v: self.G[u][v][REWARD_ATTR] for v in self.G[u]} for u in self.G if self.G[u]}

    def L(self, s):
        return s  # states are the actual props (no env involved)

    def _create_state_machine(self) -> nx.DiGraph:
        return self.G  # graph is ready after _init, no need to create
