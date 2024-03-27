import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian

# The needed pretransform to save result of EVD
class EVDTransform(object): 
    def __init__(self, norm=None):
        super().__init__()
        self.norm = norm
    def __call__(self, data):
        D, V = EVD_Laplacian(data, self.norm)
        data.eigen_values = D
        data.eigen_vectors = V.reshape(-1) # reshape to 1-d to save 
        return data

def EVD_Laplacian(data, norm=None):
    L_raw = get_laplacian(data.edge_index, normalization=norm, num_nodes=data.num_nodes)
    L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()
    D, V  = torch.linalg.eigh(L)
    return D, V

def FGSD_explicit(D, V, **params):
    if D.min() < 0:
        D = D - D.min()
    # assert D.min() >= 0, D
    if 'func' in params.keys():
        fD = D.clone()
        fD.apply_(params['func'])
    else:
        fD = D.pow(params['p'])
    assert fD.min() >= 0 
    fL = V @ torch.diag(fD) @ V.T
    ones_vector=torch.ones(fL.size(0))
    S = torch.outer(torch.diag(fL),ones_vector) + torch.outer(ones_vector,torch.diag(fL))-2*fL
    return S

def FGSD_kNN_subgraph(data, pow=-1, k=10, q=0, cal_hops=True, max_hops=5):
    S = FGSD_explicit(data.eigen_values, data.eigen_vectors.reshape(data.num_nodes,-1), p=pow)
    ###################### top k ##################################
    if q == 0:
        start = torch.arange(data.num_nodes, device=data.edge_index.device)
        k_cloest = torch.topk(S, k, largest=False)[1]
        k_cloest = torch.cat([start.unsqueeze(-1), k_cloest], dim=-1)
        node_mask = data.edge_index.new_empty((data.num_nodes, data.num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[start.repeat_interleave(k+1), k_cloest.reshape(-1)] = True
    else:
    ###################### within a distance threshold #############
        value = torch.quantile(S, q)
        node_mask = data.edge_index.new_empty((data.num_nodes, data.num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        node_mask = (S <= value)
        node_mask.fill_diagonal_(True)
    ################################################################

    if cal_hops: # this is fast enough
        row, col = data.edge_index
        num_nodes = data.num_nodes
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=data.edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float())>0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator==-1) & next_mask] = i+1
            if hop_indicator[node_mask].min() != -1:
                break 
        return node_mask, hop_indicator
    return node_mask, None

# def extract_subgraphs(data, type='EGO', sparse=False, **params):
#     assert type in ['EGO', 'RW', 'FGSD']
#     if type == 'EGO':
#         assert params.keys() >= {'num_hops'}
#         node_mask, hop_indicator = k_hop_subgraph(data.edge_index, data.num_nodes, params['num_hops'])
#     if type == 'RW':
#         assert params.keys() >= {'walk_length', 'p', 'q', 'repeat'}
#         node_mask, hop_indicator = random_walk_subgraph(data.edge_index, data.num_nodes,
#          params['walk_length'], p=params['p'], q=params['q'], repeat=params['repeat'], cal_hops=True)
#     if type == 'FGSD':
#         assert params.keys() >= {'p', 'k'}
#         node_mask, hop_indicator = FGSD_kNN_subgraph(data, pow=params['p'], k=params['k'])

#     edge_mask = node_mask[:, data.edge_index[0]] & node_mask[:, data.edge_index[1]] # N x E dense mask matrix
#     if not sparse:
#         return node_mask, edge_mask, hop_indicator
#     else:
#         return to_sparse(node_mask, edge_mask, hop_indicator)