from itertools import product

import gym
import torch
import numpy as np
import torch_geometric as tg


class PygData(gym.spaces.Space):
    def __init__(self,
                 node_features_space: gym.spaces.Space,
                 edge_features_space: gym.spaces.Space = None,
                 max_nodes: int = 100,
                 must_have_edges: bool = False,
                 seed=None):
        # shape is None since we have multiple spaces
        super().__init__(shape=None, dtype=None, seed=seed)

        # store node features and edge features
        self.nf_space = node_features_space
        self.ef_space = edge_features_space

        self.max_nodes = max_nodes
        self.must_have_edges = must_have_edges

        # gym builtin seeding
        self.seed(seed)

    def seed(self, seed=None):
        s = super().seed(seed)
        s += self.nf_space.seed(seed)
        if self.ef_space:
            s += self.ef_space.seed(seed)

        return s

    def sample(self, mask=None):
        # sample number of nodes and edges
        num_nodes = self._np_random.randint(1, self.max_nodes + 1)  # 1 to `max_nodes` nodes
        num_edges = self._np_random.randint(0, num_nodes ** 2 + 1)  # 0 to `num_nodes ** 2` edges

        if num_edges == 0 and self.must_have_edges:
            num_edges = 1

        # sample node features
        x = np.array([self.nf_space.sample() for _ in range(num_nodes)])
        x = torch.tensor(x)  # transform to tensor type

        # sample edge features if relevant
        if self.ef_space is None:
            edge_attr = None
        else:
            edge_attr = np.array([self.ef_space.sample() for _ in range(num_edges)])
            edge_attr = torch.tensor(edge_attr) # transform to tensor type

        # sample edges
        possible_edges = np.array(list(product(range(num_nodes), range(num_nodes))))
        edge_index = possible_edges[self._np_random.choice(len(possible_edges), size=num_edges, replace=False)].T
        edge_index = torch.from_numpy(edge_index)  # transform to tensor type

        # create data object
        return tg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def contains(self, data: tg.data.Data):
        if not isinstance(data, tg.data.Data):
            return False

        if data.x.shape[0] > self.max_nodes:
            return False

        if not all(self.nf_space.contains(nf) for nf in data.x.numpy()):
            return False

        if data.edge_attr is None and self.ef_space.shape != (0,):
            # edge attr not in data but space expects edge features
            return False
        elif data.edge_attr is not None and not all(self.ef_space.contains(ef) for ef in data.edge_attr.numpy()):
            return False

        return True

    def __str__(self):
        return f'PygData(max_nodes={self.max_nodes}\n' \
               f'        nf_space={self.nf_space},\n' \
               f'        ef_space={self.ef_space})'

    def __repr__(self):
        return f'PygData(max_nodes={self.max_nodes}, nf_space={self.nf_space}, ef_space={self.ef_space})'
