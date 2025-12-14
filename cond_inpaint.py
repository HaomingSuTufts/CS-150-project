"""
Conditional graph inpainting generation script.

This version assumes the model was trained conditionally, i.e. the X-network
expects x_cond = concat([x_t, x_obs, mask_x], dim=-1) as input.

Key difference from inpaint.py (baseline):
- We DO NOT clamp x/adj at every step (that was for unconditional models).
- Conditioning is provided by the score function itself through x_cond.
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
from gdss.utils.graph_utils import (
    quantize_mol,
    adjs_to_graphs,
    node_flags,
)
from gdss.core.solver_guidance import (
    ReverseDiffusionPredictor,
    EulerMaruyamaPredictor,
    LangevinCorrector,
    NoneCorrector,
)
from gdss.utils.mol_utils import gen_mol, mols_to_smiles
from gdss.utils.logger import Logger, set_log, start_log, sample_log, check_log


def disabled_train(self, mode=True):
    return self


def disable_train(model):
    model = model.eval()
    model.train = disabled_train
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_condition_file(cond_path, device_id):
    """
    Expected format (same as inpaint.py):
      {
        "x_obs":    [B, N, F],
        "adj_obs":  [B, N, N],
        "mask_x":   [B, N] or [B, N, 1]   (1=unknown, 0=known),
        "mask_adj": [B, N, N]             (1=unknown, 0=known),
        optional "flags": [B, N, N]
      }
    """
    # Support both .npz files (with named arrays) and pickled dicts
    if cond_path.endswith(".npz"):
        npz = np.load(cond_path)
        d = {k: npz[k] for k in npz.files}
    else:
        with open(cond_path, "rb") as f:
            d = pickle.load(f)

    x_obs = torch.tensor(d["x_obs"], dtype=torch.float32, device=device_id)
    adj_obs = torch.tensor(d["adj_obs"], dtype=torch.float32, device=device_id)
    mask_x = torch.tensor(d["mask_x"], dtype=torch.float32, device=device_id)
    mask_adj = torch.tensor(d["mask_adj"], dtype=torch.float32, device=device_id)

    flags = None
    if "flags" in d and d["flags"] is not None:
        flags = torch.tensor(d["flags"], dtype=torch.float32, device=device_id)

    # Ensure adj and mask dims are compatible: collapse multi-channel adjacency
    # to a single channel if appropriate, and add/remove singleton channels
    # so adj_obs and mask_adj have shapes (B, N, N).
    def _ensure_adj_mask_compat(adj, mask):
        if adj.dim() == 3 and mask.dim() == 4:
            # mask has channel dim but adj does not: add singleton channel to adj
            adj = adj.unsqueeze(1)
        elif adj.dim() == 4 and mask.dim() == 3:
            # adj has channel dim, mask does not: add channel dim to mask
            mask = mask.unsqueeze(1)
        return adj, mask

    try:
        if adj_obs.dim() != mask_adj.dim():
            adj_obs, mask_adj = _ensure_adj_mask_compat(adj_obs, mask_adj)
    except Exception:
        pass

    # Normalize channel-dimension: prefer single-channel adj/mask (B, N, N)
    if adj_obs.dim() == 4:
        if adj_obs.shape[1] == 1:
            adj_obs = adj_obs.squeeze(1)
        else:
            adj_obs = torch.argmax(adj_obs, dim=1).float()
    if mask_adj.dim() == 4:
        if mask_adj.shape[1] == 1:
            mask_adj = mask_adj.squeeze(1)
        else:
            mask_adj = torch.min(mask_adj, dim=1)[0]

    return x_obs, adj_obs, mask_x, mask_adj, flags


def make_score_fn_x_cond(sde_x, model_x, x_obs, mask_x, continuous=True):
    """
    Returns score_fn(x, adj, flags, t) where x is x_t (current state), but
    the network is fed x_cond = [x_t, x_obs, mask_x].
    """
    model_fn = model_x

    # Ensure mask_x is (B, N, 1) for concatenation
    if mask_x.dim() == 2:
        mask_x_cat = mask_x.unsqueeze(-1)
    else:
        mask_x_cat = mask_x

    def score_fn(x, adj, flags, t):
        # Build conditional input channels
        x_cond = torch.cat([x, x_obs, mask_x_cat], dim=-1)
        raw = model_fn(x_cond, adj, flags)
        std = sde_x.marginal_prob(torch.zeros_like(adj), t)[1]
        score = -raw / std[:, None, None]
        return score

    return score_fn


def make_score_fn_adj_cond(sde_adj, model_adj, x_obs, mask_x, continuous=True):
    """
    Adjacency score also uses x_cond as node input (A-net takes x as node features).
    """
    model_fn = model_adj

    if mask_x.dim() == 2:
        mask_x_cat = mask_x.unsqueeze(-1)
    else:
        mask_x_cat = mask_x

    def score_fn(x, adj, flags, t):
        x_cond = torch.cat([x, x_obs, mask_x_cat], dim=-1)
        raw = model_fn(x_cond, adj, flags)

        std = sde_adj.marginal_prob(torch.zeros_like(adj), t)[1]
        score = -raw / std[:, None, None]
        return score

    return score_fn


def sample_inpaint_cond(
    predictor_x,
    corrector_x,
    predictor_adj,
    corrector_adj,
    init_x,
    init_adj,
    flags,
    device_id,
    eps,
    # optional “safety clamp”
    x_obs=None,
    adj_obs=None,
    mask_x=None,
    mask_adj=None,
    clamp_each_step=False,
):
    """
    Same reverse SDE loop as in inpaint.py, but conditioning is handled by
    score_fn_* (through x_cond). Optionally allow safety clamping, but default off.
    """
    with torch.no_grad():
        x = init_x.clone()
        adj = init_adj.clone()

        diff_steps = predictor_adj.sde.N
        timesteps = torch.linspace(
            predictor_adj.sde.T, eps, diff_steps, device=device_id
        )

        # Ensure shapes for optional clamp
        if clamp_each_step and mask_x is not None and mask_x.dim() == 2:
            mask_x = mask_x.unsqueeze(-1)

        for i in range(diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(adj.shape[0], device=device_id) * t

            x, x_mean = corrector_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = corrector_adj.update_fn(x, adj, flags, vec_t)

            x, x_mean = predictor_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = predictor_adj.update_fn(x, adj, flags, vec_t)

            # Optional: safety clamp (not the main conditioning mechanism)
            if clamp_each_step and (x_obs is not None) and (mask_x is not None):
                x = mask_x * x + (1.0 - mask_x) * x_obs
            if clamp_each_step and (adj_obs is not None) and (mask_adj is not None):
                adj = mask_adj * adj + (1.0 - mask_adj) * adj_obs

        return adj_mean, x_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="config name (without .yaml)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to conditional-training checkpoint",
    )
    parser.add_argument(
        "--cond_path", type=str, required=True, help="path to condition pickle"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clamp_each_step", action="store_true", help="optional safety clamp each step"
    )
    args = parser.parse_args()

    # Sample config (for sampling parameters like sampler settings)
    config = get_config(args.config, args.seed)
    load_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_id = device

    # NOTE: logging will be setup after loading the checkpoint so we can use
    # the training config embedded in the checkpoint to name the experiment

    # Load ckpt dict
    # Allow non-weights-only checkpoint load to get stored tensors including
    # the full `model_config` inside the checkpoint file.
    ckpt_dict = torch.load(args.ckpt, map_location=device_id, weights_only=False)

    # Extract training config from checkpoint so we can access `train` and
    # `data` metadata for logging and model settings. The checkpoint may store
    # it either under `model_config` (older) or `config` (newer) keys.
    if "model_config" in ckpt_dict:
        configt = ckpt_dict["model_config"]
    elif "config" in ckpt_dict:
        configt = ckpt_dict["config"]
    else:
        raise KeyError("No model_config or config found in checkpoint")
    # Align random seed
    load_seed(configt.seed)

    # Resolve model params from checkpoint; the checkpoint may store either
    # a fully specified params dict, or an integer placeholder. If it's an
    # int or None, replace with current model params from `configt`.
    params_x = ckpt_dict.get("params_x", None)
    params_adj = ckpt_dict.get("params_adj", None)
    params_x_cfg, params_adj_cfg = load_model_params(configt)
    if isinstance(params_x, int) or params_x is None:
        params_x = params_x_cfg
    if isinstance(params_adj, int) or params_adj is None:
        params_adj = params_adj_cfg

    model_x = load_model_from_ckpt(params_x, ckpt_dict["x_state_dict"], device)
    model_adj = load_model_from_ckpt(params_adj, ckpt_dict["adj_state_dict"], device)
    model_x = disable_train(model_x)
    model_adj = disable_train(model_adj)

    # EMA (optional)
    if getattr(config.sample, "use_ema", False):
        ema_x = load_ema_from_ckpt(model_x, ckpt_dict.get("ema_x"), configt.train.ema)
        ema_adj = load_ema_from_ckpt(
            model_adj, ckpt_dict.get("ema_adj"), configt.train.ema
        )
        if ema_x is not None:
            ema_x.copy_to(model_x.parameters())
        if ema_adj is not None:
            ema_adj.copy_to(model_adj.parameters())

    # SDEs
    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)

    # Load conditions
    x_obs_all, adj_obs_all, mask_x_all, mask_adj_all, flags_all = load_condition_file(
        args.cond_path, device_id
    )

    num_cond, N_cond, F_orig = x_obs_all.shape
    max_node_num = configt.data.max_node_num
    batch_size = (
        args.batch_size if args.batch_size is not None else configt.data.batch_size
    )
    batch_size = max(1, batch_size)

    num_target = (
        min(args.n_samples, num_cond) if args.n_samples is not None else num_cond
    )
    if flags_all is None:
        flags_all = node_flags(
            torch.ones(num_cond, max_node_num, max_node_num, device=device_id)
        )

    # Predictor / corrector selection
    predictor_fn = (
        ReverseDiffusionPredictor
        if config.sampler.predictor == "Reverse"
        else EulerMaruyamaPredictor
    )
    corrector_fn = (
        LangevinCorrector if config.sampler.corrector == "Langevin" else NoneCorrector
    )

    gen_smiles = []
    gen_graphs = []

    n_batches = int(np.ceil(num_target / batch_size))

    # Setup logging now that we've loaded the checkpoint and configured the model
    # configuration (configt), which contains the training metadata
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{configt.data.data}-cond-inpaint"
    logger = Logger(str(os.path.join(log_dir, f"{log_name}.log")), mode="a")
    if not check_log(log_folder_name, log_name):
        logger.log(f"{log_name}")
        start_log(logger, configt)
    sample_log(logger, config)

    logger.log(
        f"Conditional inpainting: num_target={num_target}, batch_size={batch_size}, n_batches={n_batches}"
    )

    for b in range(n_batches):
        start = b * batch_size
        end = min(num_target, (b + 1) * batch_size)
        cur_bs = end - start

        x_obs_b = x_obs_all[start:end]
        adj_obs_b = adj_obs_all[start:end]
        mask_x_b = mask_x_all[start:end]
        mask_adj_b = mask_adj_all[start:end]
        flags_b = flags_all[start:end]

        init_x = sde_x.prior_sampling(x_obs_b.shape).to(device_id)
        init_adj = sde_adj.prior_sampling_sym(adj_obs_b.shape).to(device_id)

        if mask_x_b.dim() == 2:
            mask_x_blend = mask_x_b.unsqueeze(-1)
        else:
            mask_x_blend = mask_x_b
        init_x = mask_x_blend * init_x + (1.0 - mask_x_blend) * x_obs_b
        init_adj = mask_adj_b * init_adj + (1.0 - mask_adj_b) * adj_obs_b

        # Build conditional score functions
        score_fn_x = make_score_fn_x_cond(
            sde_x, model_x, x_obs_b, mask_x_b, continuous=True
        )
        score_fn_adj = make_score_fn_adj_cond(
            sde_adj, model_adj, x_obs_b, mask_x_b, continuous=True
        )

        predictor_x = predictor_fn(
            "x", sde_x, score_fn_x, config.sample.probability_flow, guidance_args=None
        )
        predictor_adj = predictor_fn(
            "adj",
            sde_adj,
            score_fn_adj,
            config.sample.probability_flow,
            guidance_args=None,
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

        # Sample
        adj_samples, x_samples = sample_inpaint_cond(
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
            mask_x=mask_x_blend,
            mask_adj=mask_adj_b,
            clamp_each_step=args.clamp_each_step,
        )

        # Postprocess / decode (same as inpaint.py)
        if config.data.data in ["QM9", "ZINC250k"]:
            # Quantize and post-process adjacency and node features to build
            # consistent inputs for `gen_mol` (which expects tensors).
            samples_int = quantize_mol(adj_samples)
            unique_vals, counts = np.unique(samples_int, return_counts=True)

            # Ensure node features are one-hot and stable
            x_th = torch.where(x_samples > 0.5, 1, 0)
            x_th = torch.concat([x_th, 1 - x_th.sum(dim=-1, keepdim=True)], dim=-1)
            class_idx = torch.argmax(x_th, dim=-1)
            x_samples_oh = torch.nn.functional.one_hot(
                class_idx, num_classes=x_th.shape[-1]
            ).to(dtype=torch.float32)

            # Post-process adjacency into int and then to one-hot
            adj_samples_mod = torch.tensor(samples_int.copy() - 1, dtype=torch.long)
            adj_samples_mod[adj_samples_mod == -1] = 3
            adj_samples_mod = torch.clamp(adj_samples_mod, 0, 3)
            adj_onehot = torch.nn.functional.one_hot(
                adj_samples_mod, num_classes=4
            ).permute(0, 3, 1, 2)

            mols, _, mol_indices = gen_mol(
                x_samples_oh, adj_onehot, dataset=configt.data.data
            )
            smiles = mols_to_smiles(mols)
            gen_smiles.extend(smiles)
        else:
            adj_q = (adj_samples > 0).float()
            graphs = adjs_to_graphs(adj_q.detach().cpu().numpy())
            gen_graphs.extend(graphs)

        logger.log(f"[Batch {b + 1}/{n_batches}] done.")

    # Save results
    out_dir = os.path.join(log_dir, "cond_inpaint_samples")
    os.makedirs(out_dir, exist_ok=True)

    if configt.data.data in ["QM9", "ZINC250k"]:
        out_path = os.path.join(out_dir, "smiles.txt")
        with open(out_path, "w") as f:
            for s in gen_smiles:
                f.write(str(s) + "\n")
        logger.log(f"Saved SMILES to: {out_path}")
    else:
        out_path = os.path.join(out_dir, "graphs.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(gen_graphs, f)
        logger.log(f"Saved graphs to: {out_path}")

    logger.log("Done.")


if __name__ == "__main__":
    main()
