import torch
import random
import numpy as np
import yaml
from easydict import EasyDict as edict

from ..models.ScoreNetwork_A import ScoreNetworkA, ScoreNetworkACond
from ..models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH, ScoreNetworkXCond
from ..core.sde import VPSDE, VESDE, subVPSDE

from ..core.losses import get_sde_loss_fn, get_sde_loss_fn_conditioned
from ..evaluation.mmd import gaussian, gaussian_emd
from .ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_yaml_config(filepath):
    with open(filepath, "r") as file:
        return edict(yaml.safe_load(file))


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = "cpu"
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop("model_type", None)
    # Normalize model_type to a string if a nested dict/EasyDict is passed
    if not isinstance(model_type, str):
        if hasattr(model_type, "type"):
            model_type = getattr(model_type, "type")
        elif isinstance(model_type, dict):
            model_type = model_type.get("type") or model_type.get("model_type")
    if model_type == "ScoreNetworkX":
        model = ScoreNetworkX(**params_)
    elif model_type == "ScoreNetworkX_GMH":
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == "ScoreNetworkA":
        model = ScoreNetworkA(**params_)
    elif model_type == "ScoreNetworkACond":
        model = ScoreNetworkACond(**params_)
    elif model_type == "ScoreNetworkXCond":
        model = ScoreNetworkXCond(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_train.lr, weight_decay=config_train.weight_decay
    )
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config_train.lr_decay
        )

    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in ["QM9", "ZINC250k"]:
        from ..utils.data_loader_mol import dataloader

        return dataloader(config, get_graph_list)
    else:
        from ..utils.data_loader import dataloader

        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == "VP":
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == "VE":
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == "subVP":
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    loss_fn = get_sde_loss_fn(
        sde_x,
        sde_adj,
        train=True,
        reduce_mean=reduce_mean,
        continuous=True,
        likelihood_weighting=False,
        eps=config.train.eps,
    )
    return loss_fn


def load_loss_fn_conditioned(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    loss_fn = get_sde_loss_fn_conditioned(
        sde_x,
        sde_adj,
        train=True,
        reduce_mean=reduce_mean,
        continuous=True,
        likelihood_weighting=False,
        eps=config.train.eps,
    )
    return loss_fn


def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    def _get(k, side=None, default=None):
        # side: 'x' or 'adj', try top-level, then side-level
        if hasattr(config_m, k):
            return getattr(config_m, k)
        if side is not None and hasattr(config_m, side):
            side_cfg = getattr(config_m, side)
            # side_cfg might be dict-like or EasyDict
            if hasattr(side_cfg, k):
                return getattr(side_cfg, k)
            try:
                return side_cfg[k]
            except Exception:
                pass
        return default

    def _extract_type(val):
        # Return a string name for model type whether val is str, object with .type/.model_type, or dict-like
        if isinstance(val, str):
            return val
        if hasattr(val, "type"):
            return getattr(val, "type")
        if hasattr(val, "model_type"):
            return getattr(val, "model_type")
        if isinstance(val, dict):
            return val.get("type") or val.get("model_type")
        return None

    model_x_type = _extract_type(getattr(config_m, "x", None))
    if model_x_type is None:
        model_x_type = getattr(config_m, "x", None)

    if model_x_type and "GMH" in str(model_x_type):
        params_x = {
            "model_type": model_x_type,
            "max_feat_num": max_feat_num,
            "depth": _get("depth", side="x"),
            "nhid": _get("nhid", side="x"),
            "num_linears": _get("num_linears", side="x"),
            "c_init": _get("c_init", side="x"),
            "c_hid": _get("c_hid", side="x"),
            "c_final": _get("c_final", side="x"),
            "adim": _get("adim", side="x"),
            "num_heads": _get("num_heads", side="x"),
            "conv": _get("conv", side="x"),
        }
    elif model_x_type == "ScoreNetworkXCond":
        orig_feat_num = _get("orig_feat_num", side="x")
        cond_feat_num = _get("cond_feat_num", side="x")
        params_x = {
            "model_type": model_x_type,
            "orig_feat_num": orig_feat_num,
            "cond_feat_num": cond_feat_num,
            "depth": _get("depth", side="x", default=_get("depth")),
            "nhid": _get("nhid", side="x", default=_get("nhid")),
        }
    else:
        params_x = {
            "model_type": config_m.x,
            "max_feat_num": max_feat_num,
            "depth": _get("depth", side="x", default=_get("depth")),
            "nhid": _get("nhid", side="x", default=_get("nhid")),
        }
    model_adj_type = _extract_type(getattr(config_m, "adj", None))
    if model_adj_type is None:
        model_adj_type = getattr(config_m, "adj", None)

    # If the adjacency model is a conditional variant, prefer cond_feat_num as its
    # input feature dimension to match the conditioned x passed to it at runtime.
    if model_adj_type and "Cond" in str(model_adj_type):
        # Prefer an explicit cond_feat_num if provided; otherwise, allow
        # an adj-side max_feat_num override (some configs set adj.max_feat_num
        # directly to the conditioned feature dim) or fallback to the global
        # max_feat_num defined for the dataset. This ensures the GNN layers
        # are constructed with the actual input feature dimension.
        cond_feat_num_adj = (
            _get("cond_feat_num", side="adj")
            or _get("max_feat_num", side="adj")
            or max_feat_num
        )
    else:
        cond_feat_num_adj = _get("max_feat_num", side="adj") or max_feat_num

    params_adj = {
        "model_type": model_adj_type,
        "max_feat_num": cond_feat_num_adj,
        "max_node_num": config.data.max_node_num,
        "nhid": _get("nhid", side="adj", default=_get("nhid", default=None)),
        "num_layers": _get(
            "num_layers",
            side="adj",
            default=_get("depth", side="adj", default=_get("depth")),
        ),
        "num_linears": _get("num_linears", side="adj"),
        "c_init": _get("c_init", side="adj"),
        "c_hid": _get("c_hid", side="adj"),
        "c_final": _get("c_final", side="adj"),
        "adim": _get("adim", side="adj"),
        "num_heads": _get("num_heads", side="adj"),
        "conv": _get("conv", side="adj"),
    }
    return params_x, params_adj


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
    path = f"./checkpoints/{config.data.data}/{config.ckpt}.pth"
    ckpt = torch.load(path, map_location=device_id, weights_only=False)
    print(f"{path} loaded")
    ckpt_dict = {
        "config": ckpt["model_config"],
        "params_x": ckpt["params_x"],
        "x_state_dict": ckpt["x_state_dict"],
        "params_adj": ckpt["params_adj"],
        "adj_state_dict": ckpt["adj_state_dict"],
    }
    if config.sample.use_ema:
        ckpt_dict["ema_x"] = ckpt["ema_x"]
        ckpt_dict["ema_adj"] = ckpt["ema_adj"]
    if return_ckpt:
        ckpt_dict["ckpt"] = ckpt
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if "module." in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")
    else:
        model = model.to(device)
    return model


def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ["degree", "cluster", "orbit", "spectral"]
    kernels = {
        "degree": gaussian_emd,
        "cluster": gaussian_emd,
        "orbit": gaussian,
        "spectral": gaussian_emd,
    }
    return methods, kernels
