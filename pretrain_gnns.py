from rmrl.nn.pretraining import pretrain_graph_embedding
from rmrl.experiments.configurations import PRETRAINED_GNN_DIR


num_props_to_cfg = {
    # 1: dict(max_nodes=2, num_iters=5_000, batch_size=1, learning_rate=1e-3),
    # 2: dict(max_nodes=4, num_iters=5_000, batch_size=4, learning_rate=1e-3),
    # 3: dict(max_nodes=6, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 4: dict(max_nodes=8, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 5: dict(max_nodes=12, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 6: dict(max_nodes=15, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 7: dict(max_nodes=18, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 8: dict(max_nodes=18, num_iters=10_000, batch_size=8, learning_rate=1e-3),
    # 11: dict(max_nodes=20, num_iters=10_000, batch_size=16, learning_rate=1e-3)
    18: dict(max_nodes=30, num_iters=15_000, batch_size=16, learning_rate=1e-3)
}


if __name__ == '__main__':
    for num_props, cfg in num_props_to_cfg.items():
        pretrain_graph_embedding(num_props=num_props,
                                 save_dir=PRETRAINED_GNN_DIR,
                                 **cfg)

        # # automatic args
        # pretrain_graph_embedding(num_props=num_props,
        #                          max_nodes=2 * num_props,
        #                          num_iters=(num_props // 5 + 1) * 5_000,
        #                          batch_size=2 ** (num_props // 5 + 1),
        #                          learning_rate=1e-3,
        #                          save_dir=PRETRAINED_GNN_DIR)
