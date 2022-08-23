import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from gym.spaces import MultiBinary, Box
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm

from rmrl.nn.models import MultilayerGNN
from rmrl.reward_machines.potential_functions import ValueIteration
from rmrl.reward_machines.reward_machine import RMFromNX, RewardMachine
from rmrl.reward_machines.rm_env import RM_DATA_KEY, CUR_STATE_PROPS_KEY
from rmrl.utils.misc import debatch_graph_to_specific_node

from stable_baselines3.common.custom_spaces import PygData

NODE_ATTR_KEY = 'x'
EDGE_ATTR_KEY = 'edge_attr'

HIGHEST_VALUE_PATH_KEY = 'highest_value_path'

GRPT_DIR = Path('grpt')
GRPT_DS_DIR = GRPT_DIR / 'ds'
GRPT_MODELS_DIR = GRPT_DIR / 'models'


class RMGraphGenerator:
    def __init__(self, num_propositions, max_nodes, r_min=-np.inf, r_max=np.inf, batch_size=None, seed=None):
        assert max_nodes <= 2 ** num_propositions

        nf_space = MultiBinary(num_propositions)
        ef_space = Box(r_min, r_max, shape=(1,))
        self.graph_space = PygData(nf_space, ef_space, max_nodes=max_nodes, must_have_edges=True, seed=seed)

        self.batch_size = batch_size

    def sample_one(self):
        # sample from graph space
        data = self.graph_space.sample()

        # return desired type
        return RMFromNX(None, to_networkx(data, node_attrs=[NODE_ATTR_KEY], edge_attrs=[EDGE_ATTR_KEY]),
                        NODE_ATTR_KEY, EDGE_ATTR_KEY)

    def sample_batch(self):
        return [self.sample_one() for _ in range(self.batch_size)]

    def sample(self, n, verbose=False):
        out = []
        iter_range = range(n)
        if verbose:
            iter_range = tqdm(iter_range, desc='generating RMs')
        for _ in iter_range:
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


# class HVPathDataset(Dataset):
#     def __init__(self, size, test_frac, num_propositions, max_nodes, max_seq_len, gamma, seed, **gen_kwargs):
#         ds_path = get_grpt_dir(GRPT_DS_DIR, size=size, test_frac=test_frac,
#                                num_propositions=num_propositions, max_nodes=max_nodes,
#                                max_seq_len=max_seq_len, gamma=gamma, **gen_kwargs) / 'ds.pkl'
#         if ds_path.exists():
#
#
#     @staticmethod
#     def get_grpt_dir(base, **kwargs):
#         p = base
#         for k, v in kwargs.items():
#             p /= f'{k}-{v}'
#         return p


def generate_dataset(size, test_frac, num_propositions, max_nodes, max_seq_len, gamma, seed, **gen_kwargs):
    gen = RMGraphGenerator(num_propositions, max_nodes, seed=seed, **gen_kwargs)
    sample = gen.sample(size, verbose=True)
    eos_token = torch.full((num_propositions,), 0.)

    ds_dicts = []
    for rm in tqdm(sample, desc='preparing dataset'):
        state_list = list(rm.all_states)
        cur_state = state_list[np.random.choice(range(len(state_list)))]
        hv_path_from_cur_state = highest_value_path(rm, cur_state, max_seq_len, gamma=gamma)
        pyg_data = rm.to_pyg_data()  # !! do this AFTER `highest_value_path` for reward shaping

        ds_dicts.append({
            CUR_STATE_PROPS_KEY: torch.tensor(cur_state),
            HIGHEST_VALUE_PATH_KEY: pad_path(hv_path_from_cur_state, max_seq_len, eos_token),
            RM_DATA_KEY: pyg_data
        })

    train_set, test_set = train_test_split(ds_dicts, test_size=test_frac)

    save_dir = get_grpt_dir(GRPT_DS_DIR, size=size, test_frac=test_frac, num_propositions=num_propositions,
                            max_nodes=max_nodes, max_seq_len=max_seq_len, gamma=gamma, seed=seed, **gen_kwargs)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save((train_set, test_set), save_dir / 'ds.pkl')


def pad_path(path, max_seq_len, pad_value):
    required_padding = (max_seq_len - len(path))
    padded_path = path + ([pad_value] * required_padding)
    return torch.tensor(padded_path).float()
    # required_padding = (max_seq_len - len(path))
    # return torch.cat([path, torch.tensor([pad_value] * required_padding)])


def get_grpt_dir(base, **kwargs):
    p = base
    for k, v in kwargs.items():
        p /= f'{k}-{v}'
    return p


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
        node_embeddings_batch = self.gnn(graph_batch.x.float(), graph_batch.edge_index, graph_batch.edge_attr)

        return debatch_graph_to_specific_node(graph_batch, node_embeddings_batch, cur_states)

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
    def __init__(self, model: HighestValueNet):
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.history = None

    def __reset_history(self):
        self.history = {
            'epoch': [],
            'training_loss': [],
            'validation_loss': []
        }

    def train(self, dl_train, dl_val, num_epochs, learning_rate, print_every=100):
        try:
            self.__reset_history()

            optimizer = Adam(self.model.parameters(), lr=learning_rate)
            for epoch in tqdm(range(1, num_epochs + 1), position=0):
                train_loss = self.train_epoch(dl_train, optimizer)
                eval_loss = self.eval_epoch(dl_val)
                self.history['epoch'].append(epoch)
                self.history['training_loss'].append(train_loss)
                self.history['validation_loss'].append(eval_loss)

                if epoch % print_every == 0:
                    print(f'epoch   {epoch}')
                    print(f'train loss: {train_loss}')
                    print(f'eval loss:  {eval_loss}')
                    print('==================================================')
                    print()

        except KeyboardInterrupt:
            print('training interrupted by user')

        return self.history

    def train_epoch(self, dl, optimizer):
        batch_losses = []
        for batch in tqdm(dl, desc='training epoch batches', leave=False, position=1):
            batch_loss = self.train_batch(batch, optimizer)
            batch_losses.append(batch_loss)

        return np.mean(batch_losses)

    def train_batch(self, rm_batch, optimizer):
        self.model.train()
        logits = self.model(rm_batch[RM_DATA_KEY], rm_batch[CUR_STATE_PROPS_KEY], rm_batch[HIGHEST_VALUE_PATH_KEY])
        loss = self.loss_fn(logits, rm_batch[HIGHEST_VALUE_PATH_KEY])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_epoch(self, dl):
        batch_losses = []
        for batch in tqdm(dl, desc='evaluation epoch batches', leave=False, position=1):
            batch_loss = self.eval_batch(batch)
            batch_losses.append(batch_loss)

        return np.mean(batch_losses)

    def eval_batch(self, rm_batch):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(rm_batch[RM_DATA_KEY], rm_batch[CUR_STATE_PROPS_KEY], rm_batch[HIGHEST_VALUE_PATH_KEY])
            loss = self.loss_fn(logits, rm_batch[HIGHEST_VALUE_PATH_KEY])

        return loss.item()

    def save_all(self, save_dir: Union[str, os.PathLike]):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)  # overwrite previous

        # save model components
        torch.save(self.model.state_dict(), save_dir / 'full')
        torch.save(self.model.gnn.state_dict(), save_dir / 'gnn')
        torch.save(self.model.rnn.state_dict(), save_dir / 'rnn')

        # save history if one exists
        if self.history is not None:
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(self.history, f)


def pretrain_graph_embedding(num_props, max_nodes, num_iters, batch_size, learning_rate, save_dir, print_every=100,
                             seed=42):
    torch.manual_seed(seed)
    rm_gen = RMGraphGenerator(num_props, max_nodes, batch_size=batch_size, seed=seed)
    model = HighestValueNet(input_size=rm_gen.graph_space.nf_space.n)
    trainer = RMGNNTrainer(rm_gen, model)
    history = trainer.train(num_iters, learning_rate, print_every)
    trainer.save_all(save_dir)
