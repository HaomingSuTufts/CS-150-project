import torch
import numpy as np
import networkx as nx

# Nedges loss
min_nedges_sq_loss = lambda x, adj: torch.sum(adj**2, dim=(1,2))
min_nedges_abs_loss = lambda x, adj: torch.sum(torch.abs(adj), dim=(1,2))
min_nedges_oneside_abs_loss = lambda x, adj, max_edges: torch.relu((torch.sum(adj, dim=(1,2)) / 2) - max_edges)
min_nedges_oneside_sq_loss = lambda x, adj, max_edges: torch.relu((torch.sum(adj, dim=(1,2)) / 2) - max_edges)**2

min_num_edges_quantized = lambda x, adj: torch.sum((adj > 0.5).float(), dim=(1,2))
min_num_edges_oneside_quantized = lambda x, adj, max_edges: torch.where(torch.sum(adj > 0.5, dim=(1,2)) / 2 < max_edges, 0., torch.sum((adj > 0.5).float(), dim=(1,2)))



# Ntriangles loss
num_triangles_sq_loss = lambda x, adj: (1/6 * torch.diagonal(torch.matrix_power(adj, 3), dim1=1, dim2=2).sum(dim=1))**2
num_triangles_abs_loss = lambda x, adj: 1/6 * torch.diagonal(torch.matrix_power(adj, 3), dim1=1, dim2=2).sum(dim=1)
num_triangles_oneside_sq_loss = lambda x, adj, max_triangles: torch.relu((1/6 * torch.diagonal(torch.matrix_power(adj, 3), dim1=1, dim2=2)).sum(dim=1) - max_triangles)**2
num_triangles_oneside_abs_loss = lambda x, adj, max_triangles: torch.relu((1/6 * torch.diagonal(torch.matrix_power(adj, 3), dim1=1, dim2=2)).sum(dim=1) - max_triangles)

num_triangles_quantized = lambda x, adj: 1/6 * torch.diagonal(torch.matrix_power((adj > 0.5).float(), 3), dim1=1, dim2=2).sum(dim=1)
num_triangles_oneside_quantized = lambda x, adj, max_triangles: torch.where((1/6 * torch.diagonal(torch.matrix_power((adj > 0.5).float(), 3), dim1=1, dim2=2).sum(dim=1)) < max_triangles, 0., 1/6 * torch.diagonal(torch.matrix_power((adj > 0.5).float(), 3), dim1=1, dim2=2).sum(dim=1))



# Maxdegree loss
min_degree_abs_loss = lambda x, adj: torch.abs(adj).sum(dim=2).mean(dim=1)
min_degree_sq_loss = lambda x, adj: (torch.abs(adj).sum(dim=2)**2).mean(dim=1)
min_degree_oneside_abs_loss = lambda x, adj, max_degree: (torch.relu(torch.abs(adj).sum(dim=2) - max_degree)).mean(dim=1)
min_degree_oneside_sq_loss = lambda x, adj, max_degree: (torch.relu(torch.abs(adj).sum(dim=2) - max_degree)**2).mean(dim=1)

max_degree_abs = lambda x, adj: torch.max(torch.abs(adj).sum(dim=2), dim=1)[0]
max_degree_oneside = lambda x, adj, max_degree: torch.where(torch.max(torch.sum(adj, dim=2), dim=1)[0] < max_degree, 0., (torch.max(torch.sum(adj, dim=2), dim=1)[0] - max_degree)**2)

max_degree_quantized = lambda x, adj: torch.max(torch.abs((adj > 0.5).float()).sum(dim=2), dim=1)[0]
max_degree_oneside_quantized = lambda x, adj, max_degree: torch.where(torch.max(torch.sum((adj > 0.5).float(), dim=2), dim=1)[0] < max_degree, 0., (torch.max(torch.sum((adj > 0.5).float(), dim=2), dim=1)[0] - max_degree)**2)


# Force stars loss
force_stars_sq_loss = lambda x, adj, nodes_per_graph: (((torch.sum(adj, dim=(1,2)) / 2) - (nodes_per_graph - 1))**2)
force_stars_abs_loss = lambda x, adj, nodes_per_graph: torch.abs(((torch.sum(adj, dim=(1,2)) / 2) - (nodes_per_graph - 1)))
force_stars_oneside_sq_loss = lambda x, adj, nodes_per_graph: torch.relu(((torch.sum(adj, dim=(1,2)) / 2) - (nodes_per_graph - 1))**2)
force_stars_oneside_abs_loss = lambda x, adj, nodes_per_graph: torch.relu(torch.abs((torch.sum(adj, dim=(1,2)) / 2) - (nodes_per_graph - 1)))

force_stars_quantized = lambda x, adj, nodes_per_graph: (torch.sum((adj > 0.5).float(), dim=(1,2)) / 2 - (nodes_per_graph - 1))**2
# This two are not in the hyperparam tuning
force_stars_oneside_quantized_abs = lambda x, adj, nodes_per_graph: torch.where(torch.sum((adj > 0.5).float(), dim=(1,2)) / 2 < (nodes_per_graph - 1), 0., torch.sum((adj > 0.5).float(), dim=(1,2)))

force_stars_loss_kwargs = lambda flags: {'nodes_per_graph': flags.sum(1)}

def count_cycles_loss(x, adj):
    adj = (adj > 0.5).float()
    loss = []
    for i in range(adj.shape[0]):
        G = nx.from_numpy_array(adj[i].cpu().numpy())
        cycles = list(nx.simple_cycles(G))
        loss.append(float(len(cycles)))
    return torch.tensor(loss, device=adj.device)

# Valency
min_valency_sq_loss = lambda x, adj, v: ((adj.sum(dim=2) - x @ v)**2).sum(1)
min_valency_abs_loss = lambda x, adj, v: (adj.sum(dim=2) - x @ v).abs().sum(1)

min_valency_oneside_sq_loss = lambda x, adj, v: (torch.relu((adj.sum(dim=2) - x @ v))**2).sum(1)
min_valency_oneside_abs_loss = lambda x, adj, v: torch.relu((adj.sum(dim=2) - x @ v)).abs().sum(1)


# Atoms
min_atoms_sq_loss = lambda x, adj, c: ((x.sum(dim=1) - c[None,:])**2).sum(1)
min_atoms_abs_loss = lambda x, adj, c: (x.sum(dim=1) - c[None,:]).abs().sum(1)

min_atoms_oneside_sq_loss = lambda x, adj, c: (torch.relu((x.sum(dim=1) - c[None,:]))**2).sum(1)
min_atoms_oneside_abs_loss = lambda x, adj, c: torch.relu((x.sum(dim=1) - c[None,:])).abs().sum(1)


# Fair graph generation

def compute_dp2(A,Z):
    (b,g,p) = Z.shape
    p_grp = Z.sum(dim=2)
    eye = torch.eye(p).to(Z.device)[None,:,:]
    Z_til = lambda a,b: ( ((Z[:,a,:][:,:,None]*Z[:,a,:][:,None,:]) * (1-eye))/(p_grp[:,a]*(p_grp[:,a]-1))[:,None,None] - 
                          ((Z[:,a,:][:,:,None]*Z[:,b,:][:,None,:]) * (1-eye))/(p_grp[:,a]*p_grp[:,b])[:,None,None] 
                          if a!=b else torch.zeros_like(A) )
    dp = (1/(g*(g-1))) * torch.sum( torch.stack([ torch.sum( Z_til(a,b) * A , dim=(1,2))**2
                                   for a in range(g) for b in np.delete(np.arange(g),a)]) , dim=0)

    return dp


def compute_dp1(A,Z):
    (b,g,p) = Z.shape
    p_grp = Z.sum(dim=2)
    eye = torch.eye(p).to(Z.device)[None,:,:]
    Z_til = lambda a,b: ( ((Z[:,a,:][:,:,None]*Z[:,a,:][:,None,:]) * (1-eye))/(p_grp[:,a]*(p_grp[:,a]-1))[:,None,None] - 
                          ((Z[:,a,:][:,:,None]*Z[:,b,:][:,None,:]) * (1-eye))/(p_grp[:,a]*p_grp[:,b])[:,None,None] 
                          if a!=b else torch.zeros_like(A) )
    dp = (1/(g*(g-1))) * torch.sum( torch.stack([ torch.abs (torch.sum( Z_til(a,b) * A, dim=(1,2) ) )
                                   for a in range(g) for b in np.delete(np.arange(g),a)]) , dim=0)

    return dp

def compute_nodedp2(A,Z):
    (b,g,p) = Z.shape
    p_grp = Z.sum(dim=2)
    eye = torch.eye(p).to(Z.device)[None,:,:]
    Z_til = lambda a,i: torch.sum(torch.stack([eye[:,i,:][:,:,None]*(Z[:,a,:]*(1-eye[:,i,:]))[:,None,:]/p_grp[:,a][:,None,None] - 
                                eye[:,i,:][:,:,None]*(Z[:,b,:]*(1-eye[:,i,:]))[:,None,:]/p_grp[:,b][:,None,None] 
                                if a!=b else torch.zeros_like(A) for b in range(g)]),dim=0)
    dp = (1/(p*g*(g-1)**2)) * torch.sum( torch.stack([ 
        torch.sum( Z_til(a,i) * A, dim=(1,2) )**2
        for a in range(g) for i in range(p) ]) , dim=0)

    return dp

def compute_nodedp1(A,Z):
    (b,g,p) = Z.shape
    p_grp = Z.sum(dim=2)
    eye = torch.eye(p).to(Z.device)[None,:,:]
    Z_til = lambda a,i: torch.sum(torch.stack([eye[:,i,:][:,:,None]*(Z[:,a,:]*(1-eye[:,i,:]))[:,None,:]/p_grp[:,a][:,None,None] - 
                                eye[:,i,:][:,:,None]*(Z[:,b,:]*(1-eye[:,i,:]))[:,None,:]/p_grp[:,b][:,None,None] 
                                if a!=b else torch.zeros_like(A) for b in range(g)]),dim=0)
    dp = (1/(p*g*(g-1)**2)) * torch.sum( torch.stack([ 
        torch.abs( torch.sum( Z_til(a,i) * A , dim=(1,2)) )
        for a in range(g) for i in range(p) ]) , dim=0)

    return dp


# Incomplete graph loss

def partially_obs_sq_loss(x, adj, idx_obs, true_adj):
    output_tensor = torch.zeros(adj.shape[0], device=adj.device)
    return output_tensor.scatter_add(0, idx_obs[0], (adj[idx_obs] - true_adj[idx_obs])**2)

def partially_obs_abs_loss(x, adj, idx_obs, true_adj):
    output_tensor = torch.zeros(adj.shape[0], device=adj.device)
    return output_tensor.scatter_add(0, idx_obs[0], torch.abs(adj[idx_obs] - true_adj[idx_obs]))

def partially_obs_diff_loss(x, adj, idx_obs, true_adj):
    output_tensor = torch.zeros(adj.shape[0], device=adj.device)
    return output_tensor.scatter_add(0, idx_obs[0], (adj[idx_obs] != true_adj[idx_obs]).float())
