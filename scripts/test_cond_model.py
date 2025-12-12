import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from gdss.utils.loader import load_yaml_config, load_model_params, load_model_optimizer

config = load_yaml_config("config/train_qm9_cond.yaml")
params_x, params_adj = load_model_params(config)
print("params_x max_feat_num:", params_x.get("max_feat_num"))
print("params_adj max_feat_num:", params_adj.get("max_feat_num"))

# Instantiate adj model and show first DenseGCNConv weight shape
model_adj, _, _ = load_model_optimizer(params_adj, config.train, "cpu")
found = False
for name, module in model_adj.named_modules():
    if module.__class__.__name__ == "DenseGCNConv":
        print("Found DenseGCNConv with weight shape", tuple(module.weight.shape))
        found = True
        break
print("Found DenseGCNConv:", found)

# Quick forward pass with dummy tensors matching expected shapes
import torch

B = 1
N = config.data.max_node_num
X = torch.randn(B, N, params_adj["max_feat_num"])
Adj = torch.randn(B, N, N)
try:
    # Inspect inner shapes in the first attention head for debugging
    layer0 = model_adj.layers[0]
    attn0 = layer0.attn[0]
    Q = attn0.gnn_q(X, Adj)
    K = attn0.gnn_k(X, Adj)
    V = attn0.gnn_v(X, Adj)
    print("Q shape", Q.shape)
    print("K shape", K.shape)
    print("V shape", V.shape)
    from math import sqrt

    dim_split = attn0.attn_dim // attn0.num_heads
    Q_ = torch.cat(Q.split(dim_split, 2), 0)
    K_ = torch.cat(K.split(dim_split, 2), 0)
    print("Q_ shape", Q_.shape)
    print("K_ shape", K_.shape)
    out = model_adj(X, Adj, flags=None)
    print("Forward pass OK. Output shape:", out.shape)
except Exception as e:
    import traceback

    traceback.print_exc()
    print("Forward pass failed with exception:", type(e).__name__, e)
