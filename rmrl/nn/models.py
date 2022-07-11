import gym
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.loader.dataloader import Collater


collator = Collater(None, None)

ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU
}


class RMFeatureExtractorSB(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, gnn_agg=torch.mean):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        #TODO make parameter
        obs_out_features = 32
        rm_out_features = 32

        extractors = {}

        self._output_size = 0
        self._gnn_agg = gnn_agg

        # group reward machine information
        rm_spaces = {}
        for key, subspace in observation_space.spaces.items():
            # TODO make generic networks as input
            if key == 'obs':
                extractors[key] = MLP(in_features=subspace.shape[0], hidden_dims=[32, 32],
                                      out_features=obs_out_features)
                self._output_size += obs_out_features
            elif key.endswith('graph'):  # rm graph
                extractors[key] = MultilayerGCN(subspace.shape[0], rm_out_features)
                self._output_size += rm_out_features
            elif key.endswith('cur_state') or key.endswith('cur_props'):
                extractors[key] = nn.Identity()  # cur does not use an extractor
                self._output_size += subspace.shape[0]
            else:
                raise ValueError(f'bad observation key {key}')

        super().__init__(observation_space, features_dim=self._output_size)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key.endswith('graph'):
                graph_data = observations[key]
                out = extractor(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                out = self._gnn_agg(out)
            else:  # key.startswith('rm')
                obs_data = observations[key]
                out = extractor(obs_data)
            encoded_tensor_list.append(out)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=-1)

    @property
    def output_size(self):
        return self._output_size


class RMPolicyNet(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Space):
        super().__init__()

        self.feature_extractor = RMFeatureExtractor(observation_space)

        if isinstance(action_space, gym.spaces.Box):
            assert len(action_space.shape) == 1, 'only supports 1d actions'
            out_features = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            out_features = action_space.n
        else:
            raise TypeError(f'RMPolicyNet does not support action_spaces of type `{type(action_space)}`')
        self.mlp = MLP(in_features=self.feature_extractor.output_size, hidden_dims=[64, 64], out_features=out_features)

    def forward(self, obs):
        feats = self.feature_extractor(obs)
        return self.mlp(feats)


class RMFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, gnn_agg=torch.mean):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__()

        #TODO make parameter
        obs_out_features = 32
        rm_out_features = 32

        extractors = {}

        total_concat_size = 0

        # original observation space runs through the observation feature extractor
        spaces = observation_space.spaces
        obs_space = spaces.pop('obs')
        #TODO make generic network for obs

        extractors['obs'] = MLP(in_features=obs_space.shape[0], hidden_dims=[32, 32], out_features=obs_out_features)
        total_concat_size += 32

        # group reward machine information
        rm_spaces = {}
        for key, subspace in observation_space.spaces.items():
            rm_label, data_label = key.split('_', maxsplit=1)
            rm_spaces.setdefault(rm_label, {})[data_label] = subspace

        for key, graph_space_dict in rm_spaces.items():
            num_node_features = graph_space_dict['node_features'].shape[1]
            num_edge_features = graph_space_dict['edge_features'].shape[1]
            extractors[key] = MultilayerGCN(num_node_features, rm_out_features)
            total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)
        self._output_size = total_concat_size
        self._gnn_agg = gnn_agg

    def forward(self, observations: dict):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'obs':
                out = extractor(torch.from_numpy(observations[key]).float())
            else:  # key.startswith('rm')
                # graph_dict = [key]
                nf = torch.from_numpy(observations[f'{key}_node_features'])
                ei = torch.from_numpy(observations[f'{key}_edge_index'])
                ef = torch.from_numpy(observations[f'{key}_edge_features'])
                # TODO: may need dim=1 if batched
                out = self._gnn_agg(extractor(nf, ei, ef).detach(), dim=0)
            encoded_tensor_list.append(out)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=-1)

    @property
    def output_size(self):
        return self._output_size


class MLP(nn.Module):
    """
    Multilayer perceptron with ReLU activations
    https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron
    """

    def __init__(self, in_features, hidden_dims, out_features, activation='relu', activation_kwargs=None,
                 output_activation=None):
        """
        initializes the MLP modules
        :param in_features: the number of input features
        :param hidden_dims: a list of hidden layer output features
        :param out_features: the number of output features
        """
        super().__init__()

        # select activation
        activation = ACTIVATIONS[activation]
        if activation_kwargs is None:
            activation_kwargs = {}

        # flatten dimensions
        all_dims = [in_features, *hidden_dims, out_features]

        # create layers
        layers = []
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                activation(**activation_kwargs)
            ]

        # remove last non-linearity
        layers = layers[:-1]

        # add output activation if needed
        if output_activation is not None:
            layers.append(output_activation)

        # create sequential model
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        runs a sequence of fully connected layers with ReLU non-linearities.
        :param x: a batch of vectors of shape (batch_size, in_features)
        """
        return self.fc_layers(x)


class MultilayerGCN(nn.Module):
    """A multilayer graph convolutional network"""

    def __init__(self, input_dim, output_dim, hidden_dims=16):
        """
        create a multilayer GCN
        :param input_dim: input node vector size
        :param output_dim: output node vector size
        :param hidden_dims: the size (or list of sizes) of the hidden layer outputs
        """
        super(MultilayerGCN, self).__init__()

        # always view hidden dims as a colleciton
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # create hidden layers
        self.layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            # create layer modules
            gcn = GCNConv(input_dim, hidden_dim, normalize=False)
            drp = nn.Dropout()
            act = nn.ReLU()

            # save layer modules together
            self.layers.append((gcn, drp, act))

            # register modules
            self.add_module(f'conv_{i}', gcn)
            self.add_module(f'dropout_{i}', drp)
            self.add_module(f'activation_{i}', act)

            # current hidden dim is the next input dim
            input_dim = hidden_dim

        # add final conv layer
        final_gcn = GCNConv(input_dim, output_dim, normalize=False)
        self.add_module(f'conv_{len(self.layers)}', final_gcn)
        self.layers.append((final_gcn, lambda x: x, lambda x: x))  # use ID func to omit dropout and activation

    def forward(self, x, edge_index, edge_weight=None):
        """
        multilayer GCN forward pass
        :param graph_data: torch_geometric.data.Data format
        """
        for gcn, drp, act in self.layers:
            x = act(drp(gcn(x.float(), edge_index, edge_weight)))
        return x
