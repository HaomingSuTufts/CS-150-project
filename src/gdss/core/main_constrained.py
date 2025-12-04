import argparse
import os
import yaml
import torch
import networkx as nx
import numpy as np
from easydict import EasyDict as edict
from datetime import datetime
from tqdm import trange

from parsers.config import get_config
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import (
    load_ckpt, load_data, load_seed, load_model_from_ckpt,
    load_ema_from_ckpt, load_sde, load_yaml_config
)
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, mask_adjs, mask_x
from solver_guidance import ReverseDiffusionPredictor, EulerMaruyamaPredictor, LangevinCorrector, NoneCorrector
from losses import get_score_fn
from utils.plot import save_graph_list, plot_graphs_list

from prodigy.project_bisection import drifted_project

from utils.loader import load_data, load_seed, load_eval_settings
from evaluation.stats import eval_graph_list

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

def get_timestring():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Graph Generation with Constraints")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., ego_small, community_small)")
    parser.add_argument('--constraint', type=str, required=True, help="Constraint type (e.g., nedges, degree, ntriangles)")
    parser.add_argument('--method', type=str, required=True, help="Method type (prodigy, loss, greedy, zero, unconstrained)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--device', type=str, default='cpu', help="Device")
    args = parser.parse_args()

    assert args.method in ['prodigy', 'loss', 'greedy', 'zero', 'unconstrained'], "Invalid method type"
    assert args.constraint in ['nedges', 'degree', 'ntriangles', 'force_stars'], "Invalid constraint type"

    # Load configuration
    config = get_config('sample_' + args.dataset, args.seed)

    if args.method in ['greedy', 'zero', 'loss']:
        guidance_config = load_yaml_config(f'config_guidance/constrained/{args.constraint}/{args.method}.yaml')
        guidance_args = edict({'method': args.method, 'obj': guidance_config['obj'], **guidance_config[args.dataset]})
        if guidance_config['obj'] == 'adj':
            guidance_args_adj = guidance_args
            guidance_args_x = None
        elif guidance_config['obj'] == 'x':
            guidance_args_x = guidance_args
            guidance_args_adj = None
    else:
        guidance_args_x = None
        guidance_args_adj = None

    # Set up device
    # device = load_device()
    if ',' in args.device:
        device = args.device.split(',')
    else:
        device = args.device
    config.device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    # Load data and models
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']
    train_graph_list, test_graph_list = load_data(configt, get_graph_list=True)
    config.train_graph_list = train_graph_list

    n_edges_test = np.array([g.number_of_edges() for g in test_graph_list])
    n_triangles_test = np.array([sum(list(nx.triangles(g).values())) for g in test_graph_list])
    max_degrees_test = np.array([max([x[1] for x in g.degree()]) for g in test_graph_list])

    max_edges = np.percentile(n_edges_test, 10)
    max_triangles = np.percentile(n_triangles_test, 10)
    if max_triangles == 0 and args.dataset != "grid":
        max_triangles = 3
    max_degree = np.percentile(max_degrees_test, 10)

    constraint_param_map = {'nedges': max_edges, 'ntriangles': max_triangles, 'degree': max_degree}

    satisfies_nedges = lambda g: g.number_of_edges() <= max_edges
    satisfies_degree = lambda g: max([x[1] for x in g.degree()]) <= max_degree
    satisfies_ntriangles = lambda g: sum(list(nx.triangles(g).values())) <= max_triangles
    satisfies_force_stars = lambda g: nx.is_isomorphic(g, nx.star_graph(g.number_of_nodes()-1))

    satisfies_fn = eval('satisfies_' + args.constraint)

    # Set up logging
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{args.dataset}-{args.constraint}-{args.method}"
    logger = Logger(str(os.path.join(log_dir, f'{log_name}.log')), mode='a')

    if not check_log(log_folder_name, log_name):
        logger.log(f'{log_name}')
        start_log(logger, configt)
        train_log(logger, configt)
    sample_log(logger, config)

    model_x = load_model_from_ckpt(ckpt_dict['params_x'], ckpt_dict['x_state_dict'], device)
    model_adj = load_model_from_ckpt(ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], device)

    model_x = disable_train(model_x)
    model_adj = disable_train(model_adj)

    if config.sample.use_ema:
        ema_x = load_ema_from_ckpt(model_x, ckpt_dict['ema_x'], configt.train.ema)
        ema_adj = load_ema_from_ckpt(model_adj, ckpt_dict['ema_adj'], configt.train.ema)
        ema_x.copy_to(model_x.parameters())
        ema_adj.copy_to(model_adj.parameters())

    if args.method == "prodigy":
        if args.constraint == 'force_stars':
            constraint_param_map = {'ntriangles': 0}
            args.constraint = 'ntriangles'
        constraint_config_path = f'prodigy/configs/{args.constraint}'
        constraint_config = load_yaml_config(f'{constraint_config_path}/constraint.yaml')
        method_config = load_yaml_config(f'{constraint_config_path}/method.yaml')
        constraint_config.params[-1] = constraint_param_map[args.constraint]

    # Logging
    logger.log(get_timestring())
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Constraint: {args.constraint}")
    logger.log(f"Method: {args.method}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Device: {device}")
    logger.log("Statistics satisfied by 10 percent of test graphs")
    logger.log(f"Max edges: {max_edges}")
    logger.log(f"Max triangles: {max_triangles}")
    logger.log(f"Max degree: {max_degree}")
    if args.method in ['greedy', 'zero', 'loss']:
        logger.log(f"Guidance args: {guidance_args}")
    elif args.method == 'prodigy':
        logger.log(f"Constraint config: {constraint_config}")
        logger.log(f"Method config: {method_config}")
    logger.log("-"*100)

    # TODO: Unify seeds
    print(f'GEN SEED: {config.sample.seed}')
    load_seed(config.sample.seed)

    # Load sampling function and SDEs
    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)

    max_node_num  = configt.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if configt.data.data in ['QM9', 'ZINC250k']:
        shape_x = (10000, max_node_num, configt.data.max_feat_num)
        shape_adj = (10000, max_node_num, max_node_num)
    else:
        shape_x = (configt.data.batch_size, max_node_num, configt.data.max_feat_num)
        shape_adj = (configt.data.batch_size, max_node_num, max_node_num)

    continuous = True
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    # Sampling
    predictor_fn = ReverseDiffusionPredictor if config.sampler.predictor == 'Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if config.sampler.corrector == 'Langevin' else NoneCorrector

    predictor_x = predictor_fn('x', sde_x, score_fn_x, config.sample.probability_flow, guidance_args_x)
    predictor_adj = predictor_fn('adj', sde_adj, score_fn_adj, config.sample.probability_flow, guidance_args_adj)
    
    corrector_x = corrector_fn('x', sde_x, score_fn_x, config.sampler.snr, config.sampler.scale_eps, config.sampler.n_steps)
    corrector_adj = corrector_fn('adj', sde_adj, score_fn_adj, config.sampler.snr, config.sampler.scale_eps, config.sampler.n_steps)

    with torch.no_grad():
        # Initialize samples
        x = predictor_x.sde.prior_sampling(shape_x).to(device_id)
        adj = predictor_adj.sde.prior_sampling_sym(shape_adj).to(device_id)
        flags = init_flags(train_graph_list, config).to(device_id)

        x = mask_x(x, flags)
        adj = mask_adjs(adj, flags)
        diff_steps = sde_adj.N
        timesteps = torch.linspace(predictor_adj.sde.T, config.sample.eps, diff_steps, device=config.device_id)

        # Reverse diffusion process
        for t in trange(len(timesteps), desc="[Sampling]", position=1, leave=False):
            vec_t = torch.ones(shape_adj[0], device=config.device_id) * timesteps[t]
            x, _ = corrector_x.update_fn(x, adj, flags, vec_t)
            adj, _ = corrector_adj.update_fn(x, adj, flags, vec_t)
            x, _ = predictor_x.update_fn(x, adj, flags, vec_t)
            adj, _ = predictor_adj.update_fn(x, adj, flags, vec_t)

            if args.method == 'prodigy':
                x, adj = drifted_project(x, adj, i=t, diff_steps=diff_steps, constraint_config=constraint_config, method_config=method_config)
                # x_mean, adj_mean = drifted_project(x_mean, adj_mean, i=t, diff_steps=diff_steps, constraint_config=constraint_config, method_config=method_config)

    samples_int = quantize(adj)
    gen_graph_list = adjs_to_graphs(samples_int, True)

    # Save results
    results_dir = os.path.join("results", log_name)
    os.makedirs(results_dir, exist_ok=True)

    g_satisfy = [satisfies_fn(g) for g in gen_graph_list]
    n_satisfy = sum(g_satisfy)

    test_graph_list = [g for g in test_graph_list if satisfies_fn(g)]

    methods, kernels = load_eval_settings(config.data.data)
    if len(test_graph_list) > 0 and len(gen_graph_list) > 0:
        results_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
    else:
        results_dict = {name: -1. for name in methods}

    mmd_names = ['degree', 'cluster', 'orbit'] # 'spectral'
    mmd_avg = np.array([results_dict[f'{name}'] for name in mmd_names]).mean()

    mmd_names_all = ['degree', 'cluster', 'orbit', 'spectral']
    mmd_avg_all = np.array([results_dict[f'{name}'] for name in mmd_names_all]).mean()

    # Save metrics
    results_dict.update({
        "num_graphs": len(gen_graph_list),
        "avg_num_edges": np.mean([g.number_of_edges() for g in gen_graph_list]),
        "avg_num_nodes": np.mean([g.number_of_nodes() for g in gen_graph_list]),
        "mmd_avg": mmd_avg,
        "mmd_avg_all": mmd_avg_all,
        "pct_satisfy": n_satisfy / len(gen_graph_list)
    })
    logger.log("Results")
    for key, value in results_dict.items():
        logger.log(f'{key}: {value}')
        results_dict[key] = float(value)

    with open(os.path.join(results_dir, "metrics.yaml"), "w") as f:
        yaml.dump(results_dict, f)

    # Save sample plots
    plot_graphs_list(gen_graph_list, title=f"samples.png", save_dir=log_name)

    save_graph_list(log_folder_name, log_name, gen_graph_list)


if __name__ == "__main__":
    main()