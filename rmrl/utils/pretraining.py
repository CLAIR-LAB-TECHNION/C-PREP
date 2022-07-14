from itertools import product, count

import torch
from torch import nn

import numpy as np

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from gym.utils import seeding
from gym.spaces import MultiDiscrete, Box


class RMGraphGenerator:
    def __init__(self, num_propositions, max_nodes, r_min=-np.inf, r_max=np.inf, batch_size=None, to_nx=False,
                 seed=None):
        assert max_nodes <= 2 ** num_propositions

        self.nf_space = MultiDiscrete([2] * num_propositions)
        self.ef_space = Box(r_min, r_max, shape=(1,))
        self.max_nodes = max_nodes

        self.batch_size = batch_size
        self.to_nx = to_nx

        # self.__possible_nodes = np.array(list(product([0., 1.], repeat=num_propositions)))

        self.__rng = None
        self.seed(seed)

    def seed(self, seed=None):
        self.__rng, _ = seeding.np_random(seed)

    def sample_one(self):
        # sample number of nodes
        num_nodes = self.__rng.integers(1, self.max_nodes)
        num_edges = self.__rng.integers(0, num_nodes ** 2)

        # sample node features
        x = np.array([self.nf_space.sample() for _ in range(num_nodes)], dtype=float)
        # nodes_idx = self.__rng.choice(len(self.__possible_nodes), size=num_nodes)
        # x = self.__possible_nodes[nodes_idx]

        # sample edge features
        edge_attr = np.array([self.ef_space.sample() for _ in range(num_edges)])

        # sample edges
        possible_edges = np.array(list(product(range(num_nodes), range(num_nodes))))
        edge_index = possible_edges[self.__rng.choice(len(possible_edges), size=num_edges, replace=False)].T

        # transform to tensor types
        x = torch.tensor(x)
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.from_numpy(edge_index)

        # create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # return desired type
        if self.to_nx:
            return to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
        else:
            return data

    def sample_batch(self):
        return [self.sample_one() for _ in self.batch_size]

    def sample(self, n):
        out = []
        for _ in range(n):
            if self.batch_size is None:
                out.append(self.sample_one())
            else:
                out.append(self.sample_batch())

        return out

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size is None:
            return self.sample_one()
        else:
            return self.sample_batch()
