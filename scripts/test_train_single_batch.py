import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.getcwd())  # ensure project root is importable to load cond_train
from gdss.parsers.config import get_config
from gdss.utils.loader import (
    load_model_params,
    load_model_optimizer,
    load_sde,
    load_data,
    load_seed,
)
from gdss.core.losses import get_sde_loss_fn_conditioned
from gdss.utils.ema import ExponentialMovingAverage

import torch

# Load config and trim it for quick test
config = get_config("train_qm9_cond", 42)
config.train.num_epochs = 1
config.data.batch_size = 2

# Load random seed
load_seed(config.seed)

# Load models and data
train_loader, test_loader = load_data(config, get_graph_list=False)
params_x, params_adj = load_model_params(config)
model_x, opt_x, sched_x = load_model_optimizer(params_x, config.train, "cpu")
model_adj, opt_adj, sched_adj = load_model_optimizer(params_adj, config.train, "cpu")

sde_x = load_sde(config.sde.x)
sde_adj = load_sde(config.sde.adj)
loss_fn = get_sde_loss_fn_conditioned(
    sde_x,
    sde_adj,
    train=True,
    reduce_mean=config.train.reduce_mean,
    eps=config.train.eps,
)

# Fetch a single batch and do a forward and backward
for x_batch, adj_batch in train_loader:
    x = x_batch.to("cpu")
    adj = adj_batch.to("cpu")

    from cond_train import sample_masks

    node_mask, mask_adj = sample_masks(x, adj, keep_ratio=0.5)

    loss_x, loss_adj = loss_fn(model_x, model_adj, x, adj, node_mask, mask_adj)
    loss = loss_x + loss_adj
    print("Loss:", loss.item())
    loss.backward()
    print("Backward OK")
    break
