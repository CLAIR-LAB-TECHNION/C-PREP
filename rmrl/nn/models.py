import gym
import torch
import torch_scatter
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, GATConv, GATv2Conv, TransformerConv, Sequential
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rmrl.reward_machines.rm_env import ORIG_OBS_KEY, CUR_STATE_PROPS_KEY, RM_DATA_KEY
from rmrl.utils.misc import debatch_graph_to_specific_node

from typing import Type, Dict, Union

GnnWithEdgeAttr = Type[Union[GINEConv, GATConv, GATv2Conv, TransformerConv]]

ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU
}


def ignore_state_mean(graphs_batch, node_embeddings_batch, cur_state_batch):
    # mean of embeddings over each graph in batch while ignoring the current state
    return torch_scatter.scatter_mean(node_embeddings_batch, graphs_batch.batch, dim=-3)


def cur_state_embedding(graphs_batch, node_embeddings_batch, cur_state_batch):
    return debatch_graph_to_specific_node(graphs_batch,
                                          node_embeddings_batch,
                                          cur_state_batch)


class RMFeatureExtractorSB(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, ofe_identity=False, ofe_hidden_dims=32, ofe_out_dim=32,
                 gnn_type: GnnWithEdgeAttr = GATConv, gnn_hidden_dims=32, gnn_out_dim=32,
                 gnn_agg=cur_state_embedding, embed_cur_state=False, pretrained_gnn_path=None):
        extractors = {}

        self._output_size = 0
        self._gnn_agg = gnn_agg

        # observations feature extractor
        if ORIG_OBS_KEY in observation_space.spaces:
            if ofe_identity:
                extractors[ORIG_OBS_KEY] = nn.Identity()
                self._output_size += observation_space.spaces[ORIG_OBS_KEY].shape[0]
            else:
                extractors[ORIG_OBS_KEY] = MLP(in_features=observation_space.spaces[ORIG_OBS_KEY].shape[0],
                                               hidden_dims=ofe_hidden_dims,
                                               out_features=ofe_out_dim)
                self._output_size += ofe_out_dim

        # cur state
        if embed_cur_state and CUR_STATE_PROPS_KEY in observation_space.spaces:
            extractors[CUR_STATE_PROPS_KEY] = nn.Identity()
            self._output_size += observation_space.spaces[CUR_STATE_PROPS_KEY].shape[0]

        # group reward machine information
        if RM_DATA_KEY in observation_space.spaces:
            assert CUR_STATE_PROPS_KEY in observation_space.spaces, 'must receive cur state'
            rm_space = observation_space.spaces[RM_DATA_KEY]

            # TODO GNN type as parameter
            extractors[RM_DATA_KEY] = MultilayerGNN(gnn_class=gnn_type,
                                                    input_dim=rm_space.nf_space.shape[-1],
                                                    output_dim=gnn_out_dim,
                                                    e_dim=rm_space.ef_space.shape[-1],
                                                    hidden_dims=gnn_hidden_dims)

            # load pre-trained and freeze weights
            if pretrained_gnn_path:
                extractors[RM_DATA_KEY].load_state_dict(torch.load(pretrained_gnn_path))
                for p in extractors[RM_DATA_KEY].parameters():
                    p.requires_grad = False

            self._output_size += gnn_out_dim

        # create module and save extractors as module dict
        super().__init__(observation_space, features_dim=self._output_size)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: Dict[str, Union[torch.Tensor, Data]]):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == RM_DATA_KEY:
                # extract batch
                graphs_batch = observations[RM_DATA_KEY]
                cur_state_batch = observations[CUR_STATE_PROPS_KEY]

                # make graph batch
                node_embeddings = extractor(graphs_batch.x, graphs_batch.edge_index, graphs_batch.edge_attr)

                out = self._gnn_agg(graphs_batch, node_embeddings, cur_state_batch)
            else:  # key is not RM
                obs_data = observations[key]
                out = extractor(obs_data)

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

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

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


class MultilayerGNN(nn.Module):
    GNN_INPUT_STR = 'x, edge_index'
    GNN_INPUT_STR_WITH_EDGE_ATTR = 'x, edge_index, edge_attr'
    GNN_FN_STR = f'{GNN_INPUT_STR} -> x'
    GNN_FN_STR_WITH_EDGE_ATTR = f'{GNN_INPUT_STR_WITH_EDGE_ATTR} -> x'
    NODES_ONLY_FN = 'x -> x'

    def __init__(self, gnn_class: GnnWithEdgeAttr, input_dim, output_dim, e_dim, hidden_dims=16, **gnn_kwargs):
        super().__init__()

        # always view hidden dims as a collection
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # GINE has different input
        if gnn_class == GINEConv:
            def gnn_class(in_dim, h_dim, edge_dim, **kwargs):
                mlp_f = MLP(in_dim, [h_dim] * 2, h_dim)
                return GINEConv(mlp_f.fc_layers, edge_dim=edge_dim, **kwargs)

        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([(gnn_class(input_dim, hidden_dim, edge_dim=e_dim, **gnn_kwargs),
                            self.GNN_FN_STR_WITH_EDGE_ATTR),
                           (nn.Dropout(), self.NODES_ONLY_FN),
                           nn.ReLU(inplace=True)])

            # current hidden dim is the next input dim
            input_dim = hidden_dim

        layers.append((gnn_class(input_dim, output_dim, edge_dim=e_dim, **gnn_kwargs),
                       self.GNN_FN_STR_WITH_EDGE_ATTR))

        self.gnn_layers = Sequential(self.GNN_INPUT_STR_WITH_EDGE_ATTR, layers)

    def forward(self, x, edge_index, edge_attr):
        return self.gnn_layers(x, edge_index, edge_attr)
