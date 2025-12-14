#!/usr/bin/env python3
"""
Helper script to build conditional inputs for graph inpainting.

It:
  * loads the dataset using the same config as training,
  * samples full graphs,
  * picks a random subset of nodes to KEEP (the observed subgraph),
  * constructs masks that clamp those nodes/edges during sampling,
    * saves everything into a .npz file. Note: adjacency and mask arrays are
    * collapsed to single-channel (N, N) format for compatibility with the
    * main data loader / generation scripts.

Mask convention (very important, must match inpaint_generation.py):

    mask_x   = 0 -> node is OBSERVED (clamped to x_obs)
              1 -> node is FREE / to be INPAINTED

    mask_adj = 0 -> edge is OBSERVED (clamped to adj_obs)
              1 -> edge is FREE / to be INPAINTED

We clamp like:
    x   = mask_x   * x   + (1 - mask_x)   * x_obs
    adj = mask_adj * adj + (1 - mask_adj) * adj_obs
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add src to path so we can import internal modules from project root
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gdss.parsers.config import get_config
from gdss.utils.loader import load_data, load_seed
from gdss.utils.graph_utils import node_flags


def build_masks_for_sample(
    x_sample: torch.Tensor,
    adj_sample: torch.Tensor,
    keep_ratio: float,
    min_keep: int = 1,
):
    """
    Given a single graph (x_sample, adj_sample), build:
        x_obs, adj_obs, mask_x, mask_adj

    x_sample:  (N, F)
    adj_sample: (C, N, N)   (QM9/ZINC style multi-channel adjacency)

    Strategy:
      * infer which nodes are "real" from x_sample (non-zero rows),
      * randomly choose a subset of real nodes to keep (observed),
      * clamp all edges between kept nodes,
      * everything else is left free to be inpainted.
    """
    x_np = x_sample.detach().cpu().numpy()
    adj_np = adj_sample.detach().cpu().numpy()

    N = x_np.shape[0]
    C = adj_np.shape[0]

    # Real nodes: any node whose feature vector is not all zeros
    real_nodes = np.abs(x_np).sum(axis=-1) > 0
    real_idx = np.where(real_nodes)[0]
    num_real = len(real_idx)

    if num_real == 0:
        # Completely empty graph; skip it
        return None

    # How many nodes to keep as context
    k_keep = max(min_keep, int(np.ceil(keep_ratio * num_real)))
    k_keep = min(k_keep, num_real)

    # Randomly choose kept nodes among real ones
    kept_idx = np.random.choice(real_idx, size=k_keep, replace=False)
    kept_idx = np.sort(kept_idx)
    kept_set = set(kept_idx.tolist())

    # Observed graph is just the original full graph
    x_obs = x_np.copy()
    adj_obs = adj_np.copy()

    # Mask for nodes: 0 on kept (observed), 1 elsewhere (including padded nodes)
    mask_x = np.ones((N, 1), dtype=np.float32)
    mask_x[kept_idx, 0] = 0.0

    # Mask for edges: 0 on edges between two kept nodes; 1 otherwise
    mask_adj = np.ones((C, N, N), dtype=np.float32)
    for i in kept_idx:
        for j in kept_idx:
            mask_adj[:, i, j] = 0.0
            mask_adj[:, j, i] = 0.0

    return x_obs, adj_obs, mask_x, mask_adj


def main():
    parser = argparse.ArgumentParser(
        description="Build .npz conditional inputs for graph inpainting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_qm9",
        help="Config name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to build conditions from",
    )
    parser.add_argument(
        "--num_conditions",
        type=int,
        default=128,
        help="Number of conditional graphs to generate",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.5,
        help="Fraction of real nodes to KEEP as observed subgraph (0<r<=1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inpaint_conditions.npz",
        help="Output .npz filename",
    )

    args = parser.parse_args()

    # Load config and seed
    config = get_config(args.config, args.seed)
    load_seed(config.seed)

    # Load data loaders
    train_loader, test_loader = load_data(config, get_graph_list=False)
    loader = train_loader if args.split == "train" else test_loader

    x_obs_list = []
    adj_obs_list = []
    mask_x_list = []
    mask_adj_list = []

    # We will also need N, C to build flags later
    max_node_num = config.data.max_node_num

    num_collected = 0
    for batch in loader:
        # train.py uses: for batch_idx, (x, adj) in enumerate(pbar): ...
        # so we stick to that interface
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_batch, adj_batch = batch[0], batch[1]
        else:
            raise RuntimeError(
                "Unexpected batch format. Expected (x, adj); "
                "please check data_loader_mol / data_loader."
            )

        # x_batch: (B, N, F), adj_batch: (B, C, N, N) or (B, N, N)
        B = x_batch.shape[0]

        # Normalize adj to (B, C, N, N)
        if adj_batch.dim() == 3:
            # assume (B, N, N) -> add a singleton channel
            adj_batch = adj_batch.unsqueeze(1)

        for b in range(B):
            if num_collected >= args.num_conditions:
                break

            x_sample = x_batch[b]  # (N, F)
            adj_sample = adj_batch[b]  # (C, N, N)

            out = build_masks_for_sample(
                x_sample, adj_sample, keep_ratio=args.keep_ratio
            )
            if out is None:
                continue

            x_obs, adj_obs, mask_x, mask_adj = out

            # Optional: ensure shapes align with config.data.max_node_num
            N = x_obs.shape[0]
            if N != max_node_num:
                # If the dataset uses a different padding size than config,
                # you can pad or truncate here. For now, we enforce equality:
                raise ValueError(
                    f"Sample N={N} != config.data.max_node_num={max_node_num}. "
                    f"Please adapt padding logic in make_inpaint_conditions.py."
                )

            # Store x as-is (B, N, F)
            x_obs_list.append(x_obs.astype(np.float32))

            # Convert adjacency to single-channel (B, N, N) for compatibility
            # with data loader and inpainting scripts which expect integer
            # adjacency matrices (0..C-1) rather than one-hot channel stacks.
            # adj_obs currently is (C, N, N) --- collapse by argmax across C.
            if adj_obs.ndim == 3:
                if adj_obs.shape[0] == 1:
                    adj_obs_simple = adj_obs[0]
                else:
                    adj_obs_simple = adj_obs.argmax(axis=0)
            else:
                # already 2D (N, N)
                adj_obs_simple = adj_obs
            adj_obs_list.append(adj_obs_simple.astype(np.float32))

            mask_x_list.append(mask_x.astype(np.float32))

            # mask_adj was built as (C, N, N) with same 0/1 across channels for
            # each observed edge; collapse to single-channel (N, N) also.
            if mask_adj.ndim == 3:
                mask_adj_simple = np.min(mask_adj, axis=0)
            else:
                mask_adj_simple = mask_adj
            mask_adj_list.append(mask_adj_simple.astype(np.float32))

            num_collected += 1

        if num_collected >= args.num_conditions:
            break

    if num_collected == 0:
        raise RuntimeError("No valid graphs collected for inpainting conditions.")

    x_obs_arr = np.stack(x_obs_list, axis=0)  # (B, N, F)
    adj_obs_arr = np.stack(adj_obs_list, axis=0)  # (B, N, N)
    mask_x_arr = np.stack(mask_x_list, axis=0)  # (B, N, 1)
    mask_adj_arr = np.stack(mask_adj_list, axis=0)  # (B, N, N)

    B = x_obs_arr.shape[0]
    N = x_obs_arr.shape[1]

    # Build flags, following the unconditional generation logic:
    # init_flags = node_flags(torch.ones(batch_size, max_node_num, max_node_num))
    # Here, we just assume all nodes are "present" in the dense grid.
    adj_ones = torch.ones(B, N, N, dtype=torch.float32)
    flags_t = node_flags(adj_ones)  # whatever shape graph_utils defines
    flags_arr = flags_t.detach().cpu().numpy()

    np.savez(
        args.output,
        x_obs=x_obs_arr,
        adj_obs=adj_obs_arr,
        mask_x=mask_x_arr,
        mask_adj=mask_adj_arr,
        flags=flags_arr,
    )

    print(f"Saved {num_collected} inpainting conditions to {args.output}")
    print(f"x_obs:   {x_obs_arr.shape}")
    print(f"adj_obs: {adj_obs_arr.shape}")
    print(f"mask_x:  {mask_x_arr.shape}")
    print(f"mask_adj:{mask_adj_arr.shape}")
    print(f"flags:   {flags_arr.shape}")


if __name__ == "__main__":
    main()
