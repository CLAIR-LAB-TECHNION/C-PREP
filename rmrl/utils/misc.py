import hashlib
from itertools import chain, combinations

import torch


def powerset(iterable, max_subset_len=None):
    # have iterable as a list
    lst = list(iterable)

    if max_subset_len is None:
        max_subset_len = len(lst) + 1

    return chain.from_iterable(combinations(lst, r) for r in range(max_subset_len))


def split_pairs(lst):
    return [
        (lst[i], lst[i + 1])
        for i in range(0, len(lst), 2)
    ]


def uniqify_samples(samples):
    """
    make a unique valued list in a repeatable way.
    taken from:
    https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    :param samples:
    :return:
    """
    value_set = set()
    return [s for s in samples             # sample is kept iff:
            if not (s in value_set or      # sample value was has not yet been seen.
                    value_set.add(s))]     # if sample wasn't seen, value is added (returns None <==> False)


def sha3_hash(item):
    return hashlib.sha3_256(str(item).encode()).hexdigest()


def debatch_graph_to_specific_node(graph_batch, node_embeddings_batch, specific_node_batch):
    nodes_per_graph = [graph_batch.x[graph_batch.batch == i]
                       for i in range(graph_batch.num_graphs)]
    embeddings_per_graph = [node_embeddings_batch[graph_batch.batch == i]
                            for i in range(graph_batch.num_graphs)]

    out = []
    for n, e, s, in zip(nodes_per_graph, embeddings_per_graph, specific_node_batch):
        cur_state_msk = n == s
        cur_state_feature_count = torch.sum(cur_state_msk, dim=-1)
        cur_state_idx_mask = cur_state_feature_count == s.shape[0]
        embedding = e[cur_state_idx_mask]
        out.append(embedding.squeeze())

    return torch.stack(out)


def repr_all_public(obj: object):
    vars_join = ', '.join(f'{k}={v}'
                          for k, v in vars(obj).items()
                          if not k.startswith(f'_{obj.__class__.__name__}'))
    return f'{obj.__class__.__name__}({vars_join})'
