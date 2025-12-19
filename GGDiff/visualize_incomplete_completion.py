#!/usr/bin/env python
"""
Create the visualization of the imcomplete graph completion
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.gridspec import GridSpec
import argparse


def load_experiment_data(dataset, method, obj_obs):
   
    dataset_lower = dataset.lower()
    
    pkl_path = f"samples/pkl/{dataset.upper()}/test/{dataset_lower}-incomplete-{obj_obs}-{method}.pkl"
    with open(pkl_path, 'rb') as f:
        gen_graphs = pickle.load(f)
    
    with open(f'data/{dataset_lower}_test_nx.pkl', 'rb') as f:
        test_graphs = pickle.load(f)
    
    results_path = f"results/{dataset_lower}-incomplete-{obj_obs}-{method}/metrics.pkl"
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    
    idx_observed = (
        data['idx_observed'][0],
        data['idx_observed'][1],
        data['idx_observed'][2]
    )
    idx = data['idx']
    
    return gen_graphs, test_graphs, idx_observed, idx


def get_observed_edges_and_nonedges(gt_graph, idx_obs_sample, obj_obs='entries'):
    
    adj = nx.adjacency_matrix(gt_graph).todense()
    n = gt_graph.number_of_nodes()
    
    i_idx = idx_obs_sample[1]  
    j_idx = idx_obs_sample[2]  
    
    observed_edges = []
    observed_nonedges = []
    
    for i, j in zip(i_idx, j_idx):
        if i >= n or j >= n:
            continue
        if i >= j:  
            continue
            
        if adj[i, j] > 0:  
            observed_edges.append((int(i), int(j)))
        else:  
            observed_nonedges.append((int(i), int(j)))
    
    return observed_edges, observed_nonedges


def check_edge_preservation(gt_graph, gen_graph, observed_edges, observed_nonedges):
    gen_edges = set(gen_graph.edges())
    gen_edges_bidirectional = set()
    for u, v in gen_edges:
        gen_edges_bidirectional.add((min(u,v), max(u,v)))
    
    preserved_edges = []
    not_preserved_edges = []
    
    for u, v in observed_edges:
        edge = (min(u,v), max(u,v))
        if edge in gen_edges_bidirectional:
            preserved_edges.append((u, v))
        else:
            not_preserved_edges.append((u, v))
    
    preserved_nonedges = []
    not_preserved_nonedges = []
    
    for u, v in observed_nonedges:
        edge = (min(u,v), max(u,v))
        if edge not in gen_edges_bidirectional:
            preserved_nonedges.append((u, v))
        else:
            not_preserved_nonedges.append((u, v))
    
    return preserved_edges, not_preserved_edges, preserved_nonedges, not_preserved_nonedges


def visualize_incomplete_graph_completion(sample_idx, dataset='qm9', methods=['loss', 'greedy', 'zero'], 
                                          obj_obs='entries', output_dir='visualizations/incomplete_comparison'):

    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    for method in methods:
        try:
            gen_graphs, test_graphs, idx_observed, idx = load_experiment_data(dataset, method, obj_obs)
            all_data[method] = {
                'gen_graphs': gen_graphs,
                'test_graphs': test_graphs,
                'idx_observed': idx_observed,
                'idx': idx
            }
        except Exception as e:
            print(f"Warning: Could not load data for {method}: {e}")
            continue
    
    if not all_data:
        print("Error: No data loaded")
        return
    
    first_method = list(all_data.keys())[0]
    test_graphs = all_data[first_method]['test_graphs']
    idx = all_data[first_method]['idx']
    
    test_idx = idx[sample_idx]
    gt_graph = test_graphs[test_idx]
    
    print(f"Visualizing sample {sample_idx} (test graph {test_idx})")
    print(f"Ground truth: {gt_graph.number_of_nodes()} nodes, {gt_graph.number_of_edges()} edges")
    
    n_methods = len(all_data)
    fig = plt.figure(figsize=(15, 4 * (n_methods + 1)))
    gs = GridSpec(n_methods + 1, 1, figure=fig, hspace=0.3)
    
    pos = nx.spring_layout(gt_graph, seed=42, k=1.5, iterations=50)
    
    node_labels = nx.get_node_attributes(gt_graph, 'label')
    if not node_labels:
        node_labels = {i: str(i) for i in gt_graph.nodes()}
    
    # ==========================================
    # 1. Ground Truth graph
    # ==========================================
    ax_gt = fig.add_subplot(gs[0])
    
    nx.draw_networkx_edges(gt_graph, pos, ax=ax_gt, edge_color='lightgray', 
                           width=2, alpha=0.5)
    
    nx.draw_networkx_nodes(gt_graph, pos, ax=ax_gt, node_color='lightblue', 
                           node_size=800, edgecolors='black', linewidths=2)
    
    nx.draw_networkx_labels(gt_graph, pos, node_labels, ax=ax_gt, 
                           font_size=12, font_weight='bold')
    
    ax_gt.set_title(f'Ground Truth Graph\n({gt_graph.number_of_nodes()} nodes, {gt_graph.number_of_edges()} edges)', 
                    fontsize=14, fontweight='bold', pad=10)
    ax_gt.axis('off')
    ax_gt.margins(0.15)
    
    # ==========================================
    # 2. Make generated graphs with edge preservation
    # ==========================================
    for method_idx, method in enumerate(methods):
        if method not in all_data:
            continue
        
        ax = fig.add_subplot(gs[method_idx + 1])
        
        gen_graph = all_data[method]['gen_graphs'][sample_idx]
        idx_observed = all_data[method]['idx_observed']
        
        mask = idx_observed[0] == sample_idx
        sample_obs_idx = (
            idx_observed[0][mask],
            idx_observed[1][mask],
            idx_observed[2][mask]
        )
        
        observed_edges, observed_nonedges = get_observed_edges_and_nonedges(
            gt_graph, sample_obs_idx, obj_obs
        )
        
        preserved_edges, not_preserved_edges, preserved_nonedges, not_preserved_nonedges = \
            check_edge_preservation(gt_graph, gen_graph, observed_edges, observed_nonedges)
        
        print(f"\n{method.upper()}:")
        print(f"  Generated: {gen_graph.number_of_nodes()} nodes, {gen_graph.number_of_edges()} edges")
        print(f"  Observed edges: {len(observed_edges)}, Preserved: {len(preserved_edges)}, Not preserved: {len(not_preserved_edges)}")
        print(f"  Observed non-edges: {len(observed_nonedges)}, Preserved: {len(preserved_nonedges)}, Not preserved: {len(not_preserved_nonedges)}")
        
        if gen_graph.number_of_edges() > 0:
            nx.draw_networkx_edges(gen_graph, pos, ax=ax, edge_color='lightgray', 
                                   width=1.5, alpha=0.3, style='dotted')
        
        if preserved_edges:
            nx.draw_networkx_edges(gen_graph, pos, preserved_edges, ax=ax, 
                                   edge_color='green', width=3, alpha=0.8, 
                                   style='solid', label='Preserved edges')
        
        if not_preserved_edges:
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(gt_graph.nodes())
            temp_graph.add_edges_from(not_preserved_edges)
            nx.draw_networkx_edges(temp_graph, pos, not_preserved_edges, ax=ax, 
                                   edge_color='red', width=3, alpha=0.8, 
                                   style='solid', label='Not preserved edges')
        
        if preserved_nonedges:
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(gt_graph.nodes())
            temp_graph.add_edges_from(preserved_nonedges)
            nx.draw_networkx_edges(temp_graph, pos, preserved_nonedges, ax=ax, 
                                   edge_color='green', width=2, alpha=0.6, 
                                   style='dashed', label='Preserved non-edges')
        
        if not_preserved_nonedges:
            nx.draw_networkx_edges(gen_graph, pos, not_preserved_nonedges, ax=ax, 
                                   edge_color='red', width=2, alpha=0.6, 
                                   style='dashed', label='Not preserved non-edges')
        
        gen_node_labels = nx.get_node_attributes(gen_graph, 'label')
        if not gen_node_labels:
            gen_node_labels = {i: str(i) for i in gen_graph.nodes()}
        
        nodes_to_draw = list(gen_graph.nodes())
        if nodes_to_draw:
            node_pos = {n: pos[n] for n in nodes_to_draw if n in pos}
            nx.draw_networkx_nodes(gen_graph, node_pos, nodelist=nodes_to_draw,
                                   ax=ax, node_color='lightblue', 
                                   node_size=800, edgecolors='black', linewidths=2)
            nx.draw_networkx_labels(gen_graph, node_pos, 
                                   {n: gen_node_labels.get(n, str(n)) for n in nodes_to_draw},
                                   ax=ax, font_size=12, font_weight='bold')
        
        accuracy = len(preserved_edges) / len(observed_edges) if observed_edges else 0
        ax.set_title(f'{method.upper()} Method\n'
                     f'({gen_graph.number_of_nodes()} nodes, {gen_graph.number_of_edges()} edges, '
                     f'Edge accuracy: {accuracy:.1%})', 
                     fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        ax.margins(0.15)
    
    legend_elements = [
        mpatches.Patch(color='green', label='Observed edges (preserved)', alpha=0.8),
        mpatches.Patch(color='red', label='Observed edges (not preserved)', alpha=0.8),
        mpatches.Patch(facecolor='white', edgecolor='green', label='Observed non-edges (preserved)', 
                      linestyle='--', linewidth=2),
        mpatches.Patch(facecolor='white', edgecolor='red', label='Observed non-edges (not preserved)', 
                      linestyle='--', linewidth=2),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=11, framealpha=0.9)
    
    save_path = os.path.join(output_dir, f'incomplete_completion_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {save_path}")
    plt.close()


def create_multiple_samples_comparison(dataset='qm9', methods=['loss', 'greedy', 'zero'],
                                       obj_obs='entries', n_samples=5,
                                       output_dir='visualizations/incomplete_comparison'):

    print(f"="*70)
    print(f"Creating Incomplete Graph Completion Visualizations")
    print(f"="*70)
    print(f"Dataset: {dataset.upper()}")
    print(f"Observation type: {obj_obs}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Number of samples: {n_samples}")
    print(f"Output directory: {output_dir}")
    print(f"="*70)
    
    for i in range(n_samples):
        print(f"\n[{i+1}/{n_samples}] Processing sample {i}...")
        try:
            visualize_incomplete_graph_completion(
                sample_idx=i,
                dataset=dataset,
                methods=methods,
                obj_obs=obj_obs,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Completed! Visualizations saved to: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize incomplete graph completion experiments"
    )
    parser.add_argument('--dataset', type=str, default='qm9',
                       help='Dataset name (qm9, enzymes)')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['loss', 'greedy', 'zero'],
                       help='Methods to compare')
    parser.add_argument('--obj_obs', type=str, default='entries',
                       choices=['entries', 'edges'],
                       help='Observation type')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, 
                       default='visualizations/incomplete_comparison',
                       help='Output directory')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='Specific sample index to visualize (if set, ignores n_samples)')
    
    args = parser.parse_args()
    
    if args.sample_idx is not None:
        visualize_incomplete_graph_completion(
            sample_idx=args.sample_idx,
            dataset=args.dataset,
            methods=args.methods,
            obj_obs=args.obj_obs,
            output_dir=args.output_dir
        )
    else:
        create_multiple_samples_comparison(
            dataset=args.dataset,
            methods=args.methods,
            obj_obs=args.obj_obs,
            n_samples=args.n_samples,
            output_dir=args.output_dir
        )
