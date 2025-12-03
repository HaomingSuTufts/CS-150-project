import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from scipy.stats import chi2
import concurrent.futures
import itertools

# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


# -------- Create initial node features --------
def init_features(init, adjs=None, nfeat=10):

    if init=='zeros':
        feature = torch.zeros((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='ones':
        feature = torch.ones((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='deg':
        feature = adjs.sum(dim=-1).to(torch.long)
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            print(feature.max().item())
            raise NotImplementedError(f'max_feat_num mismatch')
    else:
        raise NotImplementedError(f'{init} not implemented')

    flags = node_flags(adjs)

    return mask_x(feature, flags)


# -------- Sample initial flags tensor from the training graph set --------
def init_flags(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])

    return flags


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


# -------- Quantize generated graphs --------
def quantize(adjs, thr=0.5):
    adjs_ = torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))
    return adjs_


# -------- Quantize generated molecules --------
# adjs: 32 x 9 x 9
def quantize_mol(adjs):                         
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def adjs_to_graphs(adjs, is_cuda=False):
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list

def return_com_node(list_com, i):
    for j in range(len(list_com)):
        if i in list_com[j]:
            return j

def compute_group_assignments(g, n_com):
    if type(g) != nx.Graph:
        g = g.to_networkx()
    communities = nx.community.greedy_modularity_communities(g, weight=None, best_n=n_com) # greedy_modularity_communities louvain_communities
    idxs_com = [return_com_node(communities, i) for i in g.nodes()]

    return torch.nn.functional.one_hot(torch.tensor(idxs_com), num_classes=n_com).float()


def count_across_community_edges(adjs, Zs):
    # Get the community assignments (0 or 1) for each node
    communities = torch.argmax(Zs, dim=1)  # Shape: (b, n)

    # Create a mask for across-community edges
    # Compare the community of each pair of nodes
    community_diff = communities.unsqueeze(-1) != communities.unsqueeze(-2)  # Shape: (b, n, n)

    # Count edges where nodes belong to different communities
    across_community_edges = adjs * community_diff  # Mask adjacency matrix with across-community edges
    across_community_edge_counts = across_community_edges.sum(dim=(1, 2)) / 2 # Sum over n x n for each graph

    return across_community_edge_counts

def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
            )
    return count / float(len(G_list))

def est_p_intra_inter(graph, communities=None):
    """
    Compute the probabilities of connection between communities and within communities.

    Parameters:
    - graph: A NetworkX graph.
    - communities: A tuple where each element is a list of nodes in a community.

    Returns:
    - p_between: Probability of connection between communities.
    - p_within: Probability of connection within communities.
    """
    if communities is None:
        communities = nx.community.louvain_communities(graph)
    # Compute probability of connection between communities
    edges_between = 0
    possible_edges_between = 0
    for com1, com2 in itertools.combinations(communities, 2):
        edges_between += sum(1 for u, v in itertools.product(com1, com2) if graph.has_edge(u, v))
        possible_edges_between += len(com1) * len(com2)
    p_between = edges_between / possible_edges_between if possible_edges_between > 0 else 0

    # Compute probability of connection within communities
    p_within_list = []
    for community in communities:
        subgraph = graph.subgraph(community)
        num_edges = subgraph.number_of_edges()
        num_possible_edges = len(community) * (len(community) - 1) / 2
        p_within_list.append(num_edges / num_possible_edges if num_possible_edges > 0 else 0)
    p_within = sum(p_within_list) / len(p_within_list) if p_within_list else 0

    return p_between, p_within


def is_sbm_graph(G, p_intra=None, p_inter=None, strict=True, communities=None, factor=10.):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    est_p_intra, est_p_inter = est_p_intra_inter(G, communities=communities)

    if p_intra is None or p_inter is None:
        return est_p_inter > factor * est_p_intra
    else:

        W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
        W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)
        p = 1 - chi2.cdf(abs(W), 1)
        p = p.mean()
        if strict:
            return p > 0.9  # p value < 10 %
        else:
            return p


# -------- Check if the adjacency matrices are symmetric --------
def check_sym(adjs, print_val=False):
    sym_error = (adjs-adjs.transpose(-1,-2)).abs().sum([0,1,2])
    if not sym_error < 1e-2:
        raise ValueError(f'Not symmetric: {sym_error:.4e}')
    if print_val:
        print(f'{sym_error:.4e}')


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


# -------- Create padded adjacency matrices --------
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def graphs_to_tensor(graph_list, max_node_num):
    adjs_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        adj = nx.to_numpy_array(g, nodelist=node_list)
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        adjs_list.append(padded_adj)

    del graph_list

    adjs_np = np.asarray(adjs_list)
    del adjs_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    del adjs_np

    return adjs_tensor 


def graphs_to_adj(graph, max_node_num):
    max_node_num = max_node_num

    assert isinstance(graph, nx.Graph)
    node_list = []
    for v, feature in graph.nodes.data('feature'):
        node_list.append(v)

    adj = nx.to_numpy_array(graph, nodelist=node_list)
    padded_adj = pad_adjs(adj, node_number=max_node_num)

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    del padded_adj

    return adj


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair
