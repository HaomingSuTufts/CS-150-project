#!/usr/bin/env python3
"""
Small generation script to load a trained checkpoint and create new graphs.
This script mirrors the sampling logic used by the gdss sampling scripts but is
exposed as a simple CLI at the project root for convenience.

It expects checkpoints at ./checkpoints/<DATASET>/<CKPT>.pth and uses the
sample_<dataset>.yaml config files when available.
"""

import argparse
import os
import sys
import pickle
import torch
import numpy as np

# Add src to path so we can import internal modules as packages from root
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gdss.parsers.config import get_config
from gdss.utils.loader import (
    load_model_from_ckpt,
    load_sde,
    load_seed,
    load_ema_from_ckpt,
    load_model_params,
)
from gdss.utils.graph_utils import quantize_mol, adjs_to_graphs, node_flags, quantize
from gdss.utils.plot import plot_graphs_list
from gdss.core.losses import get_score_fn
from gdss.core.solver_guidance import (
    ReverseDiffusionPredictor,
    EulerMaruyamaPredictor,
    LangevinCorrector,
    NoneCorrector,
)
from gdss.utils.mol_utils import gen_mol, mols_to_smiles, mols_to_nx
from gdss.utils.logger import Logger, set_log, start_log, sample_log, check_log


def disabled_train(self, mode=True):
    return self


def disable_train(model):
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    return model


def sample(
    predictor_x,
    corrector_x,
    predictor_adj,
    corrector_adj,
    init_x,
    init_adj,
    flags,
    device_id,
    eps,
):
    with torch.no_grad():
        x = init_x.clone()
        adj = init_adj.clone()
        diff_steps = predictor_adj.sde.N
        timesteps = torch.linspace(
            predictor_adj.sde.T, eps, diff_steps, device=device_id
        )

        for i in range(0, diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(init_adj.shape[0], device=t.device) * t

            _x = x
            x, _ = corrector_x.update_fn(x, adj, flags, vec_t)
            adj, _ = corrector_adj.update_fn(_x, adj, flags, vec_t)

            _x = x
            x, _ = predictor_x.update_fn(x, adj, flags, vec_t)
            adj, _ = predictor_adj.update_fn(_x, adj, flags, vec_t)

    return adj, x


def main():
    parser = argparse.ArgumentParser(
        description="Unconditional Graph Generation using trained checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="QM9", help="Dataset name (QM9, ZINC250k)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename (without .pth). If omitted, the config's ckpt is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for sampling (e.g. cpu, 0, 0,1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_samples", type=int, default=16, help="Number of graphs to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size to use during sampling. Defaults to n_samples or config batch_size",
    )
    args = parser.parse_args()

    args.dataset = args.dataset
    config = get_config("sample_" + args.dataset.lower(), args.seed)
    if args.ckpt is not None:
        config.ckpt = args.ckpt

    # Device parsing
    if "," in args.device:
        device = args.device.split(",")
    else:
        device = args.device
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else args.device
    config.device_id = device_id

    # Load checkpoint directly to handle naming mismatch between training and loader
    path = f"./checkpoints/{config.data.data}/{config.ckpt}.pth"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"{path} loaded")

    # Handle cases where the saved checkpoint used 'config' vs 'model_config'
    if "model_config" in ckpt:
        configt = ckpt["model_config"]
    elif "config" in ckpt:
        configt = ckpt["config"]
    else:
        raise KeyError("No model_config or config found in checkpoint")

    params_x = ckpt.get("params_x", None)
    params_adj = ckpt.get("params_adj", None)
    # In some saved checkpoints params_x/params_adj may be the integer of parameter counts
    # (train.py overwrote them later) - fall back to model config
    # Build model params in the shape expected by loader if we only got numeric values
    params_x_cfg, params_adj_cfg = load_model_params(configt)
    if isinstance(params_x, int) or params_x is None:
        params_x = params_x_cfg
    if isinstance(params_adj, int) or params_adj is None:
        params_adj = params_adj_cfg

    x_state_dict = ckpt["x_state_dict"]
    adj_state_dict = ckpt["adj_state_dict"]
    ckpt_dict = {
        "config": configt,
        "params_x": params_x,
        "x_state_dict": x_state_dict,
        "params_adj": params_adj,
        "adj_state_dict": adj_state_dict,
    }
    if config.sample.use_ema:
        ckpt_dict["ema_x"] = ckpt.get("ema_x", None)
        ckpt_dict["ema_adj"] = ckpt.get("ema_adj", None)

    # Set seeds
    load_seed(configt.seed)

    # Prepare logging
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{args.dataset}-generation"
    logger = Logger(str(os.path.join(log_dir, f"{log_name}.log")), mode="a")
    if not check_log(log_folder_name, log_name):
        logger.log(f"{log_name}")
        start_log(logger, configt)
    sample_log(logger, config)

    # Load models
    model_x = load_model_from_ckpt(
        ckpt_dict["params_x"], ckpt_dict["x_state_dict"], device
    )
    model_adj = load_model_from_ckpt(
        ckpt_dict["params_adj"], ckpt_dict["adj_state_dict"], device
    )

    model_x = disable_train(model_x)
    model_adj = disable_train(model_adj)

    if config.sample.use_ema:
        ema_x = load_ema_from_ckpt(model_x, ckpt_dict["ema_x"], configt.train.ema)
        ema_adj = load_ema_from_ckpt(model_adj, ckpt_dict["ema_adj"], configt.train.ema)
        ema_x.copy_to(model_x.parameters())
        ema_adj.copy_to(model_adj.parameters())

    # Setup SDEs and sampling functions
    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)

    max_node_num = configt.data.max_node_num
    batch_size = (
        args.batch_size
        if args.batch_size
        else min(args.n_samples, configt.data.batch_size)
    )
    batch_size = max(1, batch_size)

    # batch-level shapes (constructed per-batch later)

    # For unconditional generation, assume all potential nodes are present
    init_flags = node_flags(torch.ones(batch_size, max_node_num, max_node_num))

    # Define sampling functions
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=True)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=True)
    predictor_fn = (
        ReverseDiffusionPredictor
        if config.sampler.predictor == "Reverse"
        else EulerMaruyamaPredictor
    )
    corrector_fn = (
        LangevinCorrector if config.sampler.corrector == "Langevin" else NoneCorrector
    )

    predictor_x = predictor_fn(
        "x", sde_x, score_fn_x, config.sample.probability_flow, guidance_args=None
    )
    predictor_adj = predictor_fn(
        "adj", sde_adj, score_fn_adj, config.sample.probability_flow, guidance_args=None
    )
    corrector_x = corrector_fn(
        "x",
        sde_x,
        score_fn_x,
        config.sampler.snr,
        config.sampler.scale_eps,
        config.sampler.n_steps,
    )
    corrector_adj = corrector_fn(
        "adj",
        sde_adj,
        score_fn_adj,
        config.sampler.snr,
        config.sampler.scale_eps,
        config.sampler.n_steps,
    )

    n_batches = int(np.ceil(args.n_samples / batch_size))
    gen_graphs = []
    gen_smiles = []

    for b in range(n_batches):
        cur_batch = min(batch_size, args.n_samples - b * batch_size)
        shape_x_b = (cur_batch, max_node_num, configt.data.max_feat_num)
        shape_adj_b = (cur_batch, max_node_num, max_node_num)

        init_x = sde_x.prior_sampling(shape_x_b).to(device)
        init_adj = sde_adj.prior_sampling_sym(shape_adj_b).to(device)
        init_flags_b = init_flags[:cur_batch].to(device)

        adj_samples, x_samples = sample(
            predictor_x,
            corrector_x,
            predictor_adj,
            corrector_adj,
            init_x,
            init_adj,
            init_flags_b,
            device_id,
            config.sample.eps,
        )

        if configt.data.data in ["QM9", "ZINC250k"]:
            # Debug: inspect adj_samples stats before quantization
            try:
                adj_stats = adj_samples.detach().cpu().numpy()
                nan_count = np.isnan(adj_stats).sum()
                logger.log(
                    f"Adj float stats: min={np.nanmin(adj_stats)}, max={np.nanmax(adj_stats)}, mean={np.nanmean(adj_stats)}, nan_count={int(nan_count)}"
                )
            except Exception:
                logger.log("Could not compute float adjacency stats")
            samples_int = quantize_mol(adj_samples)
            # Log quantization statistics for debugging
            unique_vals, counts = np.unique(samples_int, return_counts=True)
            try:
                sval = dict(zip(unique_vals.tolist(), counts.tolist()))
            except Exception:
                sval = str(unique_vals)
            try:
                smin, smax = samples_int.min(), samples_int.max()
            except Exception:
                smin, smax = None, None
            logger.log(
                f"Quantized adjacency unique values: {sval}; dtype={samples_int.dtype}; min={smin}, max={smax}"
            )
            try:
                logger.log(f"Sample entries (first sample):\n{samples_int[0, :, :]}\n")
            except Exception:
                logger.log("Could not print sample adjacency matrix")
            # Ensure node features are one-hot by thresholding and then taking argmax
            x_th = torch.where(x_samples > 0.5, 1, 0)
            x_th = torch.concat([x_th, 1 - x_th.sum(dim=-1, keepdim=True)], dim=-1)
            # Fix multi-hot or ambiguous cases by taking argmax across features
            class_idx = torch.argmax(x_th, dim=-1)
            x_samples = torch.nn.functional.one_hot(
                class_idx, num_classes=x_th.shape[-1]
            ).to(dtype=torch.float32)

            # Post-process samples
            adj_samples_mod = torch.tensor(samples_int.copy() - 1, dtype=torch.long)
            adj_samples_mod[adj_samples_mod == -1] = 3
            adj_samples_mod = torch.clamp(adj_samples_mod, 0, 3)
            adj_onehot = torch.nn.functional.one_hot(
                adj_samples_mod, num_classes=4
            ).permute(0, 3, 1, 2)
            # Try generating molecules. If we get zero molecules, try a fallback
            # using the argmax approach directly on the raw network outputs (no threshold)
            gen_mols, _ = gen_mol(x_samples, adj_onehot, configt.data.data)
            if len(gen_mols) == 0:
                logger.log(
                    "No molecules generated from thresholded features; trying argmax fallback."
                )
                # Raw argmax fallback (no thresholding)
                class_idx_raw = torch.argmax(x_samples, dim=-1)
                x_arg = torch.nn.functional.one_hot(
                    class_idx_raw, num_classes=x_samples.shape[-1]
                ).to(dtype=torch.float32)
                gen_mols, _ = gen_mol(x_arg, adj_onehot, configt.data.data)
            gen_graph_list = mols_to_nx(gen_mols)
            gen_smiles_batch = mols_to_smiles(gen_mols)
            gen_smiles_batch = [smi for smi in gen_smiles_batch if len(smi)]
            logger.log(
                f"Generated molecules: {len(gen_mols)}; SMILES generated: {len(gen_smiles_batch)}"
            )
            gen_graphs.extend(gen_graph_list)
            gen_smiles.extend(gen_smiles_batch)
        else:
            samples_int = quantize(adj_samples)
            gen_graph_list = adjs_to_graphs(samples_int, True)
            gen_graphs.extend(gen_graph_list)

        logger.log(
            f"Generated batch {b + 1}/{n_batches} - total graphs: {len(gen_graphs)}"
        )

    # Save samples
    save_dir = os.path.join("samples", log_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    output_name = f"{log_name}_{config.ckpt}_{args.n_samples}.pkl"
    with open(os.path.join(save_dir, output_name), "wb") as f:
        pickle.dump({"graphs": gen_graphs, "smiles": gen_smiles}, f)
    logger.log(
        f"Saved {len(gen_graphs)} generated graphs to {os.path.join(save_dir, output_name)}"
    )

    # Save simple visuals
    plot_graphs_list(gen_graphs, title=f"samples_{args.dataset}.png", save_dir=log_name)

    if gen_smiles:
        # Save smiles list to file too
        with open(
            os.path.join(save_dir, f"{log_name}_{config.ckpt}_{args.n_samples}.smi"),
            "w",
        ) as f:
            f.write("\n".join(gen_smiles))
        logger.log(f"Saved {len(gen_smiles)} SMILES to file.")


if __name__ == "__main__":
    main()
