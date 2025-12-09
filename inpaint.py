"""
Graph inpainting generation script.

Baseline idea:
- Use the *unconditional* GDSS diffusion model (same as generation.py).
- At sampling time, clamp a given subgraph (x_obs, adj_obs) using masks, and
  only let the unknown region diffuse.

Condition file format:
    {
        "x_obs":    float32 array [B, N, F],
        "adj_obs":  float32 or int array [B, N, N],
        "mask_x":   float32 array [B, N] or [B, N, 1] or [B, N, F]
                    (1 = unknown / to inpaint, 0 = known),
        "mask_adj": float32 array [B, N, N]
                    (1 = unknown / to inpaint, 0 = known),
        # optional:
        "flags":    float32 array [B, N, N] (node flags for padding)
    }

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


def sample_inpaint(
    predictor_x,
    corrector_x,
    predictor_adj,
    corrector_adj,
    init_x,
    init_adj,
    flags,
    device_id,
    eps,
    x_obs=None,
    adj_obs=None,
    mask_x=None,
    mask_adj=None,
):
    """
    Reverse SDE sampler with optional inpainting.

    If x_obs/adj_obs and masks are provided, we clamp the known region at every
    iteration:

        x = mask_x * x + (1 - mask_x) * x_obs
        adj = mask_adj * adj + (1 - mask_adj) * adj_obs

    where mask == 1 means "unknown (to inpaint)" and mask == 0 means "known".
    """
    with torch.no_grad():
        x = init_x.clone()
        adj = init_adj.clone()
        diff_steps = predictor_adj.sde.N
        timesteps = torch.linspace(
            predictor_adj.sde.T, eps, diff_steps, device=device_id
        )

        conditional = (
            (x_obs is not None)
            and (adj_obs is not None)
            and (mask_x is not None)
            and (mask_adj is not None)
        )

        if conditional:
            # Ensure everything is on the same device
            x_obs = x_obs.to(device_id)
            adj_obs = adj_obs.to(device_id)
            mask_x = mask_x.to(device_id)
            mask_adj = mask_adj.to(device_id)

            # Normalize mask_x shape to [B, N, 1] or [B, N, F]
            if mask_x.dim() == 2:
                mask_x = mask_x.unsqueeze(-1)

        for i in range(0, diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(init_adj.shape[0], device=t.device) * t

            # Corrector
            _x = x
            x, _ = corrector_x.update_fn(x, adj, flags, vec_t)
            adj, _ = corrector_adj.update_fn(_x, adj, flags, vec_t)

            # Predictor
            _x = x
            x, _ = predictor_x.update_fn(x, adj, flags, vec_t)
            adj, _ = predictor_adj.update_fn(_x, adj, flags, vec_t)

            # Inpainting clamp: enforce observed subgraph
            if conditional:
                x = mask_x * x + (1.0 - mask_x) * x_obs
                adj = mask_adj * adj + (1.0 - mask_adj) * adj_obs

        return adj, x


def load_condition_file(cond_path, device):
    """
    Load conditioning partial graphs + masks.

    Supports:
      - .npz: expects keys x_obs, adj_obs, mask_x, mask_adj, optional flags
      - .pkl / .pickle: expects a dict with the same keys
    """
    if cond_path.endswith(".npz"):
        data = np.load(cond_path)
        cond = {k: data[k] for k in data.files}
    else:
        with open(cond_path, "rb") as f:
            cond = pickle.load(f)

    # Convert to torch tensors
    def to_tensor(name):
        arr = cond[name]
        return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr

    x_obs = to_tensor("x_obs").float()
    adj_obs = to_tensor("adj_obs").float()
    mask_x = to_tensor("mask_x").float()
    mask_adj = to_tensor("mask_adj").float()

    flags = cond.get("flags", None)
    if flags is not None:
        flags = to_tensor("flags").float()

    # Move later to device batch-wise; here just sanity check shapes
    B, N, F = x_obs.shape
    assert adj_obs.shape[:2] == (B, N)
    assert mask_adj.shape[:2] == (B, N)

    return x_obs, adj_obs, mask_x, mask_adj, flags


def main():
    parser = argparse.ArgumentParser(
        description="Graph inpainting using unconditional GDSS diffusion model"
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
        "--cond_path",
        type=str,
        required=True,
        help="Path to condition file (.npz or .pkl) with x_obs, adj_obs, mask_x, mask_adj",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size during sampling. Defaults to min(#cond, config batch_size)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="(Optional) number of conditional graphs to inpaint; "
        "defaults to all graphs in cond file.",
    )
    args = parser.parse_args()

    # Config loading (same pattern as generation.py)
    args.dataset = args.dataset
    config = get_config("sample_" + args.dataset.lower(), args.seed)
    if args.ckpt is not None:
        config.ckpt = args.ckpt

    # Device parsing (string like "0" or "0,1" or "cpu")
    if "," in args.device:
        device = args.device.split(",")
    else:
        device = args.device
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else args.device
    config.device_id = device_id

    # Load checkpoint directly (mirrors generation.py)
    path = f"./checkpoints/{config.data.data}/{config.ckpt}.pth"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"{path} loaded")

    # Handle 'model_config' vs 'config' naming in checkpoint
    if "model_config" in ckpt:
        configt = ckpt["model_config"]
    elif "config" in ckpt:
        configt = ckpt["config"]
    else:
        raise KeyError("No model_config or config found in checkpoint")

    params_x = ckpt.get("params_x", None)
    params_adj = ckpt.get("params_adj", None)
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

    # Logging (mirrors generation.py)
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{args.dataset}-inpaint"
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
        args.batch_size if args.batch_size is not None else configt.data.batch_size
    )
    batch_size = max(1, batch_size)

    # Load conditional graphs + masks
    x_obs_all, adj_obs_all, mask_x_all, mask_adj_all, flags_all = load_condition_file(
        args.cond_path, device_id
    )
    num_cond, N_cond, F_cond = x_obs_all.shape

    if args.n_samples is not None:
        num_target = min(args.n_samples, num_cond)
    else:
        num_target = num_cond

    # Flags: if not provided, assume all nodes present
    if flags_all is None:
        flags_all = node_flags(torch.ones(num_cond, max_node_num, max_node_num))

    # Define score functions and samplers
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

    # Inpainting loop
    n_batches = int(np.ceil(num_target / batch_size))
    gen_graphs = []
    gen_smiles = []

    logger.log(
        f"Starting inpainting on {num_target} conditional graphs "
        f"(total in cond file: {num_cond}), batch_size={batch_size}"
    )

    for b in range(n_batches):
        start = b * batch_size
        end = min(num_target, (b + 1) * batch_size)
        cur_batch = end - start

        x_obs_b = x_obs_all[start:end].to(device_id)
        adj_obs_b = adj_obs_all[start:end].to(device_id)
        mask_x_b = mask_x_all[start:end].to(device_id)
        mask_adj_b = mask_adj_all[start:end].to(device_id)
        flags_b = flags_all[start:end].to(device_id)

        # Sample from prior with the same shape as observed graphs
        shape_x_b = x_obs_b.shape  # (cur_batch, N, F)
        shape_adj_b = adj_obs_b.shape  # (cur_batch, N, N)
        init_x = sde_x.prior_sampling(shape_x_b).to(device_id)
        init_adj = sde_adj.prior_sampling_sym(shape_adj_b).to(device_id)

        # Blend prior and observed graph at t=T (so known region is exactly x_obs/adj_obs)
        if mask_x_b.dim() == 2:
            mask_x_b = mask_x_b.unsqueeze(-1)
        init_x = mask_x_b * init_x + (1.0 - mask_x_b) * x_obs_b
        init_adj = mask_adj_b * init_adj + (1.0 - mask_adj_b) * adj_obs_b

        adj_samples, x_samples = sample_inpaint(
            predictor_x,
            corrector_x,
            predictor_adj,
            corrector_adj,
            init_x,
            init_adj,
            flags_b,
            device_id,
            config.sample.eps,
            x_obs=x_obs_b,
            adj_obs=adj_obs_b,
            mask_x=mask_x_b,
            mask_adj=mask_adj_b,
        )

        # --- Postprocess & decode (QM9 / ZINC vs generic) ---
        if configt.data.data in ["QM9", "ZINC250k"]:
            # Debug stats
            try:
                adj_stats = adj_samples.detach().cpu().numpy()
                nan_count = np.isnan(adj_stats).sum()
                logger.log(
                    f"[Batch {b + 1}] Adj float stats: "
                    f"min={np.nanmin(adj_stats)}, max={np.nanmax(adj_stats)}, "
                    f"mean={np.nanmean(adj_stats)}, nan_count={int(nan_count)}"
                )
            except Exception:
                logger.log(f"[Batch {b + 1}] Could not compute float adjacency stats")

            samples_int = quantize_mol(adj_samples)
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
                f"[Batch {b + 1}] Quantized adjacency unique values: {sval}; "
                f"dtype={samples_int.dtype}; min={smin}, max={smax}"
            )

            # Ensure node features are one-hot
            x_th = torch.where(x_samples > 0.5, 1, 0)
            x_th = torch.concat(
                [x_th, 1 - x_th.sum(dim=-1, keepdim=True)],
                dim=-1,
            )
            class_idx = torch.argmax(x_th, dim=-1)
            x_samples_oh = torch.nn.functional.one_hot(
                class_idx, num_classes=x_th.shape[-1]
            ).to(dtype=torch.float32)

            # Post-process adjacency
            adj_samples_mod = torch.tensor(samples_int.copy() - 1, dtype=torch.long)
            adj_samples_mod[adj_samples_mod == -1] = 3
            adj_samples_mod = torch.clamp(adj_samples_mod, 0, 3)
            adj_onehot = torch.nn.functional.one_hot(
                adj_samples_mod, num_classes=4
            ).permute(0, 3, 1, 2)

            gen_mols, _ = gen_mol(x_samples_oh, adj_onehot, configt.data.data)
            if len(gen_mols) == 0:
                logger.log(
                    "[Batch {b+1}] No molecules from thresholded features; "
                    "falling back to raw argmax."
                )
                class_idx_raw = torch.argmax(x_samples, dim=-1)
                x_arg = torch.nn.functional.one_hot(
                    class_idx_raw, num_classes=x_samples.shape[-1]
                ).to(dtype=torch.float32)
                gen_mols, _ = gen_mol(x_arg, adj_onehot, configt.data.data)

            gen_graph_list = mols_to_nx(gen_mols)
            gen_smiles_batch = mols_to_smiles(gen_mols)
            gen_smiles_batch = [smi for smi in gen_smiles_batch if len(smi)]
            logger.log(
                f"[Batch {b + 1}] Inpainted molecules: {len(gen_mols)}; "
                f"SMILES: {len(gen_smiles_batch)}"
            )
            gen_graphs.extend(gen_graph_list)
            gen_smiles.extend(gen_smiles_batch)
        else:
            samples_int = quantize(adj_samples)
            gen_graph_list = adjs_to_graphs(samples_int, True)
            gen_graphs.extend(gen_graph_list)

        logger.log(
            f"Inpainted batch {b + 1}/{n_batches} - total graphs so far: {len(gen_graphs)}"
        )

    # Save results
    save_dir = os.path.join("samples", log_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    output_name = f"{log_name}_{config.ckpt}_{num_target}.pkl"
    with open(os.path.join(save_dir, output_name), "wb") as f:
        pickle.dump({"graphs": gen_graphs, "smiles": gen_smiles}, f)
    logger.log(
        f"Saved {len(gen_graphs)} inpainted graphs to "
        f"{os.path.join(save_dir, output_name)}"
    )

    # Simple visuals
    plot_graphs_list(gen_graphs, title=f"inpaint_{args.dataset}.png", save_dir=log_name)

    if gen_smiles:
        with open(
            os.path.join(save_dir, f"{log_name}_{config.ckpt}_{num_target}.smi"),
            "w",
        ) as f:
            f.write("\n".join(gen_smiles))
        logger.log(f"Saved {len(gen_smiles)} SMILES to file.")


if __name__ == "__main__":
    main()
