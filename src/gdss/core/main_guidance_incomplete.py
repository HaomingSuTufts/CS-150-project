import argparse
import os
import torch
import pickle
import numpy as np
from easydict import EasyDict as edict
from tqdm import trange
import networkx as nx

from parsers.config import get_config
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import (
    load_ckpt,
    load_data,
    load_seed,
    load_model_from_ckpt,
    load_ema_from_ckpt,
    load_sde,
    load_yaml_config,
)
from utils.graph_utils import (
    graphs_to_tensor,
    node_flags,
    quantize,
    quantize_mol,
    adjs_to_graphs,
    is_sbm_graph,
)
from solver_guidance import (
    ReverseDiffusionPredictor,
    EulerMaruyamaPredictor,
    LangevinCorrector,
    NoneCorrector,
)
from losses import get_score_fn
from evaluation.stats import eval_graph_list
from moses.metrics.metrics import get_all_metrics
from utils.mol_utils import (
    gen_mol,
    mols_to_smiles,
    load_smiles,
    canonicalize_smiles,
    mols_to_nx,
)
from utils.plot import save_graph_list, plot_graphs_list


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
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

        for i in trange(0, diff_steps, desc="[Sampling]", position=1, leave=False):
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
        description="Incomplete Graph Generation Experiment"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., QM9, ZINC250k)"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["loss", "greedy", "zero", "unconstrained"],
        help="Method type (loss, greedy, zero, unconstrained)",
    )
    parser.add_argument(
        "--pct_obs",
        type=float,
        default=0.5,
        help="Percentage of observed entries (e.g., 0.5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (e.g., cpu, cuda:0)"
    )
    parser.add_argument(
        "--obj_obs",
        type=str,
        default="entries",
        choices=["entries", "edges"],
        help="Whether to observe entries or edges.",
    )
    args = parser.parse_args()

    # Load configuration
    args.dataset = args.dataset.lower()
    config = get_config("sample_" + args.dataset, args.seed)

    if args.method in ["greedy", "zero", "loss"]:
        guidance_config = load_yaml_config(
            f"config_guidance/incomplete/{args.obj_obs}/{args.method}.yaml"
        )
        guidance_args = edict(
            {
                "method": args.method,
                "obj": guidance_config["obj"],
                **guidance_config[args.dataset],
            }
        )
        if guidance_config["obj"] == "adj":
            guidance_args_adj = guidance_args
            guidance_args_x = None
        elif guidance_config["obj"] == "x":
            guidance_args_x = guidance_args
            guidance_args_adj = None
    else:
        guidance_args_x = None
        guidance_args_adj = None

    # Set up device
    # device = load_device()
    if "," in args.device:
        device = args.device.split(",")
    else:
        device = args.device
    config.device_id = f"cuda:{device[0]}" if isinstance(device, list) else device

    # Load checkpoint and data
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict["config"]

    load_seed(configt.seed)
    if configt.data.data in ["QM9", "ZINC250k"]:
        # train_graph_list, _ = load_data(configt, get_graph_list=True)
        with open(f"data/{configt.data.data.lower()}_test_nx.pkl", "rb") as f:
            test_graph_list = pickle.load(f)
        train_smiles, test_smiles = load_smiles(configt.data.data)
        test_smiles = canonicalize_smiles(test_smiles)
    else:
        train_graph_list, test_graph_list = load_data(configt, get_graph_list=True)

    # Prepare logging
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{args.dataset}-incomplete-{args.obj_obs}-{args.method}"
    logger = Logger(str(os.path.join(log_dir, f"{log_name}.log")), mode="a")
    if not check_log(log_folder_name, log_name):
        logger.log(f"{log_name}")
        start_log(logger, configt)
        train_log(logger, configt)
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

    # Sampling setup
    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)

    max_node_num = configt.data.max_node_num

    # Considering a batch size of 10000 for ZINC250k gives OOM error
    # batch_size = 10000 if configt.data.data in ['QM9', 'ZINC250k'] else configt.data.batch_size
    batch_size = 10000 if configt.data.data in ["QM9"] else configt.data.batch_size
    shape_x = (batch_size, max_node_num, configt.data.max_feat_num)
    shape_adj = (batch_size, max_node_num, max_node_num)

    graph_tensor = graphs_to_tensor(test_graph_list, max_node_num)
    idx = np.random.randint(0, len(test_graph_list), batch_size)
    ground_truth_adjs = graph_tensor[idx].to(device)
    init_flags_iter = node_flags(graph_tensor[idx]).to(device)

    # Mask observed entries
    if args.obj_obs == "entries":
        random_samps = torch.rand(len(test_graph_list), max_node_num, max_node_num)[
            idx
        ].to(device)
        random_samps = (random_samps + random_samps.transpose(-1, -2)) / 2
        bool_tensor = random_samps < args.pct_obs
        bool_tensor = bool_tensor & ~torch.eye(max_node_num, device=device).bool()
        idx_observed = torch.where(bool_tensor)
    elif args.obj_obs == "edges":
        idx_edges = torch.where(ground_truth_adjs != 0)
        n_edges_tot = idx_edges[0].shape[0]
        idx_obs = torch.randperm(n_edges_tot)[: int(args.pct_obs * n_edges_tot)]
        idx_observed = (
            idx_edges[0][idx_obs],
            idx_edges[1][idx_obs],
            idx_edges[2][idx_obs],
        )

    idx_observed = (
        idx_observed[0].to(device),
        idx_observed[1].to(device),
        idx_observed[2].to(device),
    )

    guidance_args_adj["loss_kwargs"] = {
        "idx_obs": idx_observed,
        "true_adj": ground_truth_adjs,
    }

    # Initialize samples
    init_x = sde_x.prior_sampling(shape_x).to(device)
    init_adj = sde_adj.prior_sampling_sym(shape_adj).to(device)

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
        "x",
        sde_x,
        score_fn_x,
        config.sample.probability_flow,
        guidance_args=guidance_args_x,
    )
    predictor_adj = predictor_fn(
        "adj",
        sde_adj,
        score_fn_adj,
        config.sample.probability_flow,
        guidance_args=guidance_args_adj,
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

    # Sampling
    adj_samples, x_samples = sample(
        predictor_x,
        corrector_x,
        predictor_adj,
        corrector_adj,
        init_x,
        init_adj,
        init_flags_iter,
        device,
        config.sample.eps,
    )

    if configt.data.data in ["QM9", "ZINC250k"]:
        samples_int = quantize_mol(adj_samples)
        x_samples = torch.where(x_samples > 0.5, 1, 0)
        x_samples = torch.concat(
            [x_samples, 1 - x_samples.sum(dim=-1, keepdim=True)], dim=-1
        )

        # Post-process samples
        adj_samples_mod = samples_int.copy() - 1
        adj_samples_mod[adj_samples_mod == -1] = 3
        adj_onehot = torch.nn.functional.one_hot(
            torch.tensor(adj_samples_mod), num_classes=4
        ).permute(0, 3, 1, 2)
        gen_mols, _, _ = gen_mol(x_samples, adj_onehot, configt.data.data)
        gen_graph_list = mols_to_nx(gen_mols)
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        # Evaluate metrics
        scores = get_all_metrics(
            gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8, test=test_smiles
        )
        scores_nspdk = eval_graph_list(
            test_graph_list, gen_graph_list, methods=["nspdk"]
        )["nspdk"]

        acc = (
            torch.tensor(samples_int).to(device)[idx_observed]
            == ground_truth_adjs[idx_observed]
        ).sum() / idx_observed[0].shape[0]

        # Log results
        logger.log(f"Accuracy: {acc.item()}")
        logger.log(f"Metrics: {scores}")
        logger.log(f"NSPDK: {scores_nspdk}")

        metrics = {"acc": acc.item(), "nspdk": scores_nspdk, "scores": scores}

    else:
        valid_fns = {
            "ego_small": lambda g: np.any(g.degree() == g.number_of_nodes() - 1),
            "community_small": lambda g: is_sbm_graph(g, factor=8.0),
        }
        eval_valid = lambda g_list: np.mean(
            [
                valid_fns[configt.data.data](g)
                if configt.data.data in valid_fns
                else -1.0
                for g in g_list
            ]
        )

        samples_int = quantize(adj_samples)
        gen_graph_list = adjs_to_graphs(samples_int, True)

        adjs = torch.zeros(
            len(gen_graph_list),
            configt.data.max_node_num,
            configt.data.max_node_num,
            device=device,
        )
        for i, G in enumerate(gen_graph_list):
            nG = G.number_of_nodes()
            adjs[i, :nG, :nG] = torch.tensor(
                nx.adjacency_matrix(G).todense(), device=device
            )
        # xs = torch.zeros (len(gen_graph_list), configt.data.max_node_num, configt.data.max_feat_num)

        acc = (
            adjs[idx_observed] == ground_truth_adjs[idx_observed]
        ).sum() / idx_observed[0].shape[0]

        pct_valid = eval_valid(gen_graph_list)
        logger.log(f"Accuracy: {acc.item()}")
        logger.log(f"% Valid samples: {pct_valid}")
        logger.log(f"Generated graphs: {len(gen_graph_list)}")

        metrics = {"acc": acc.item(), "pct_valid": pct_valid}

    # Save sample plots
    plot_graphs_list(gen_graph_list, title=f"samples.png", save_dir=log_name)

    save_graph_list(log_folder_name, log_name, gen_graph_list)

    # Save results
    results_dir = os.path.join("results", log_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(
            {
                "metrics": metrics,
                "idx_observed": (
                    idx_observed[0].cpu().numpy(),
                    idx_observed[1].cpu().numpy(),
                    idx_observed[2].cpu().numpy(),
                ),
                "idx": idx,
            },
            f,
        )


if __name__ == "__main__":
    main()
