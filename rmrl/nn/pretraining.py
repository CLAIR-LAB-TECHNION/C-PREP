import numpy as np
import torch
from gym.spaces import MultiBinary, Box
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm

from rmrl.nn.models import MultilayerGNN
from rmrl.reward_machines.potential_functions import ValueIteration
from rmrl.reward_machines.reward_machine import RMFromNX, RewardMachine
from rmrl.utils.custom_spaces import PygData

NODE_ATTR_KEY = 'x'
EDGE_ATTR_KEY = 'edge_attr'

RM_GEN_OUTPUT_PYG = 'pyg'
RM_GEN_OUTPUT_NX = 'nx'
RM_GEN_OUTPUT_RM = 'rm'
RM_GEN_OUTPUT_ALL = 'all'


class RMGraphGenerator:
    def __init__(self, num_propositions, max_nodes, r_min=-np.inf, r_max=np.inf, batch_size=None, seed=None):
        assert max_nodes <= 2 ** num_propositions

        nf_space = MultiBinary(num_propositions)
        ef_space = Box(r_min, r_max, shape=(1,))
        self.graph_space = PygData(nf_space, ef_space, max_nodes=max_nodes, seed=seed)

        self.batch_size = batch_size

    def sample_one(self):
        # sample from graph space
        data = self.graph_space.sample()

        # return desired type
        return RMFromNX(None, to_networkx(data, node_attrs=[NODE_ATTR_KEY], edge_attrs=[EDGE_ATTR_KEY]),
                        NODE_ATTR_KEY, EDGE_ATTR_KEY)

    def sample_batch(self):
        return [self.sample_one() for _ in range(self.batch_size)]

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


def highest_value_path(rm: RewardMachine, start, max_path_length, reshape=True, gamma=0.99):
    if reshape:
        pots = ValueIteration()(rm, gamma)
        rm.reshape_rewards(pots, gamma)

    path = [start]
    while len(path) < max_path_length:
        if start not in rm.delta:
            break

        next_node = max(rm.delta[start], key=lambda v: rm.delta[start][v])
        path.append(next_node)
        start = next_node

    return path


class HighestValueNet(nn.Module):
    def __init__(self, input_size, gnn_class=GATConv, gnn_hidden_size=32, gnn_layers=2, gnn_embedding_size=32,
                 decoder_layers=2, max_seq_len=100):
        super().__init__()

        self.gnn_embedding_size = gnn_embedding_size
        self.decoder_layers = decoder_layers
        self.max_seq_len = max_seq_len

        self.gnn = MultilayerGNN(
            gnn_class=gnn_class,
            input_dim=input_size,
            output_dim=gnn_embedding_size,
            edge_dim=1,  # only one edge attribute: reward
            hidden_dims=[gnn_hidden_size] * gnn_layers
        )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=gnn_embedding_size,
            num_layers=decoder_layers,
            batch_first=True,
            bidirectional=False,  # unidirectional decoder
        )

        self.hidden2output = nn.Linear(gnn_hidden_size, input_size)  # output size is the same

        self.eos_token = torch.full((input_size,), 0.)

    def forward_graph_embedding(self, graph_batch, cur_states):
        node_embeddings_scatter_batch = self.gnn(graph_batch.x.float(), graph_batch.edge_index, graph_batch.edge_attr)

        nodes_per_graph = [graph_batch.x[graph_batch.batch == i]
                           for i in range(graph_batch.num_graphs)]
        embeddings_per_graph = [node_embeddings_scatter_batch[graph_batch.batch == i]
                                for i in range(graph_batch.num_graphs)]

        out = []
        for n, e, s, in zip(nodes_per_graph, embeddings_per_graph, cur_states):
            cur_state_msk = n == s
            cur_state_feature_count = torch.sum(cur_state_msk, dim=-1)
            cur_state_idx_mask = cur_state_feature_count == s.shape[0]
            embedding = e[cur_state_idx_mask]
            out.append(embedding)

        return torch.stack(out).squeeze()

    def forward_rnn_decoder(self, encoder_output_state, tgt_seq):
        rnn_outputs, _ = self.rnn(tgt_seq, encoder_output_state)

        # we are using batch_first=True
        rnn_outputs.permute(1, 0, 2)

        return self.hidden2output(rnn_outputs)

    def forward(self, graph_batch, cur_states, tgt_seq):
        embeddings = self.forward_graph_embedding(graph_batch, cur_states)

        # we have embeddings of size H
        # we need one for each layer N (repeat)
        # reshape to (N, B, H) as required (N is batch size)
        rnn_hidden_state_per_layer = embeddings.repeat(1, self.decoder_layers).view(self.decoder_layers, -1,
                                                                                    self.gnn_embedding_size)
        rnn_outputs = self.forward_rnn_decoder(rnn_hidden_state_per_layer, tgt_seq)

        return rnn_outputs


class RMGNNTrainer:
    def __init__(self,
                 rm_gen: RMGraphGenerator,
                 model: HighestValueNet):
        self.rm_gen = rm_gen
        self.model = model

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.history = None

    def __reset_history(self):
        self.history = {
            'iteration': [],
            'training_loss': [],
            'validation_loss': []
        }

    def train(self, num_iters, learning_rate, print_every=100):
        try:
            self.__reset_history()

            optimizer = Adam(self.model.parameters(), lr=learning_rate)
            for i, (train_batch, eval_batch) in tqdm(enumerate(zip(self.rm_gen, self.rm_gen), 1), total=num_iters):
                train_loss = self.train_batch(train_batch, optimizer)
                eval_loss = self.eval_batch(eval_batch)
                self.history['iteration'].append(i)
                self.history['training_loss'].append(train_loss)
                self.history['validation_loss'].append(eval_loss)

                if i % print_every == 0:
                    print(f'iteration   {i}')
                    print(f'train loss: {train_loss}')
                    print(f'eval loss:  {eval_loss}')
                    print('==================================================')
                    print()

                if i == num_iters:
                    break

        except KeyboardInterrupt:
            print('training interrupted by user')

        return self.history

    def prep_batch(self, rm_batch):
        cur_states = []
        highest_value_paths = []
        pyg_datas = []
        for rm in rm_batch:
            state_list = list(rm.all_states)
            cur_states.append(state_list[np.random.choice(range(len(state_list)))])
            highest_value_paths.append(highest_value_path(rm, cur_states[-1], self.model.max_seq_len))
            pyg_datas.append(rm.to_pyg_data())

        pyg_batch = Batch.from_data_list(pyg_datas)
        cur_state_batch = torch.tensor(cur_states)
        highest_value_paths_batch = self.__handle_highest_value_paths(highest_value_paths)

        return pyg_batch, cur_state_batch, highest_value_paths_batch

    def __handle_highest_value_paths(self, highest_value_paths):
        max_seq_len = max(map(len, highest_value_paths))
        if max_seq_len < self.model.max_seq_len:
            max_seq_len += 1  # if all sequences terminate, make sure they have at least one end-of-seq token
        for p in highest_value_paths:
            required_padding = (max_seq_len - len(p))
            p.extend([self.model.eos_token] * required_padding)
        return torch.tensor(highest_value_paths).float()

    def train_batch(self, rm_batch, optimizer):
        self.model.train()
        pyg_datas, cur_states, paths = self.prep_batch(rm_batch)
        logits = self.model(pyg_datas, cur_states, paths)
        loss = self.loss_fn(logits, paths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_batch(self, rm_batch):
        self.model.eval()

        with torch.no_grad():
            pyg_datas, cur_states, paths = self.prep_batch(rm_batch)
            logits = self.model(pyg_datas, cur_states, paths)
            loss = self.loss_fn(logits, paths)

        return loss.item()
