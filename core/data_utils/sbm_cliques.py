import numpy as np, os
import pickle
import torch
from torch_geometric.utils.random import stochastic_blockmodel_graph
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

class CliqueSBM(InMemoryDataset):
    """
        Task 1: binary classification, whether contains clique
        Task 2: regress clique ratios
        Task 3: regress diameter
    """
    tasks = [1, 2, 3]
    splits = ['train', 'val', 'test']
    np.random.seed(0)
    num_graphs = 4000
    NUM_NODES_PER_BLOCK = 6
    NUM_BLOCKS = 10
    # INTER_PROB = 0.015
    # INTRA_PROB = 0.7
    def __init__(self, root, split, task=1, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        assert task in self.tasks
        path = os.path.join(self.processed_dir, f'task{task}_{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["generated_data_task1.pkl", "generated_data_task2.pkl", "generated_data_task3.pkl"]

    @property
    def processed_file_names(self):
        return [f'task{t}_{s}.pt' for t in self.tasks for s in self.splits]

    def download(self):
        # generate dataset
        print("Generating dataset...")
        # filename = f"{self.raw_dir}/generated_data.pkl"
        all_tasks = []
        all_tasks.append([generate_sbm_task1(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.015, 0.7, True) for _ in range(self.num_graphs//2)] 
                          + [generate_sbm_task1(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.015, 0.7, False) for _ in range(self.num_graphs//2)])
        all_tasks.append([generate_sbm_task2(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.05, 0.85, 0.1) for _ in range(self.num_graphs//4)] 
                         + [generate_sbm_task2(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.05, 0.85, 0.3) for _ in range(self.num_graphs//4)] 
                         + [generate_sbm_task2(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.05, 0.85, 0.5) for _ in range(self.num_graphs//4)] 
                         + [generate_sbm_task2(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.05, 0.85, 0.7) for _ in range(self.num_graphs//4)])
        all_tasks.append([generate_sbm_task3(self.NUM_NODES_PER_BLOCK, self.NUM_BLOCKS, 0.9) for _ in range(self.num_graphs)])

        for i, data_list in enumerate(all_tasks):
            train_idx, vali_idx = train_test_split(np.arange(self.num_graphs), test_size=0.2)
            vali_idx, test_idx =  train_test_split(vali_idx, test_size=0.5)
            splits = {'train': train_idx, 'val': vali_idx, 'test': test_idx}
            with open(self.raw_paths[i], 'wb') as f:
                pickle.dump((data_list, splits), f)
        
    def process(self):
        # Read data into huge `Data` list.
        for t in self.tasks:
            with open(self.raw_paths[t-1], 'rb') as f:
                data_list_all, splits = pickle.load(f)
            for split, idx in splits.items():
                data_list = [data_list_all[i] for i in idx]
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                data, slices = self.collate(data_list)
                torch.save((data, slices), os.path.join(self.processed_dir, f'task{t}_{split}.pt'))


def compute_block_num_edges(edges, NUM_NODES_PER_BLOCK, NUM_BLOCKS):
    num_edges = []
    mask = torch.zeros(NUM_NODES_PER_BLOCK*NUM_BLOCKS, dtype=torch.bool)
    for idx in range(NUM_BLOCKS):
        mask.fill_(0)
        mask[idx * NUM_NODES_PER_BLOCK: (idx+1) * NUM_NODES_PER_BLOCK] = 1
        block_edges = edges[:, mask[edges[0]] & mask[edges[1]]]
        num_edges.append(block_edges.size(1))
    return np.array(num_edges)

def generate_sbm_task1(NUM_NODES_PER_BLOCK, NUM_BLOCKS, INTER_PROB, INTRA_PROB, with_clique=True):
    block_sizes = np.ones(NUM_BLOCKS) * NUM_NODES_PER_BLOCK
    edge_probs = np.ones((NUM_BLOCKS, NUM_BLOCKS)) * INTER_PROB
    np.fill_diagonal(edge_probs, INTRA_PROB)

    data = Data()
    if not with_clique:
        data.y = torch.Tensor([1]).to(torch.int64)
        while True:
            edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
            has_clique = compute_block_num_edges(edge_index, NUM_NODES_PER_BLOCK, NUM_BLOCKS).max() == NUM_NODES_PER_BLOCK*(NUM_NODES_PER_BLOCK-1)
            if not has_clique and len(np.unique(edge_index)) == NUM_NODES_PER_BLOCK * NUM_BLOCKS:
                break
    else:
        data.y = torch.Tensor([0]).to(torch.int64)
        while True:
            edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
            block_num_edges = compute_block_num_edges(edge_index, NUM_NODES_PER_BLOCK, NUM_BLOCKS)
            has_clique = max(block_num_edges) == NUM_NODES_PER_BLOCK*(NUM_NODES_PER_BLOCK-1)
            if not has_clique:
                # rb_idx = np.random.randint(NUM_BLOCKS)
                rb_idx = block_num_edges.argmax() # pick up the densest one

                ### Find edges inside and outside the block
                inside, outside, others = [], [], []
                for e1, e2 in edge_index.T.numpy():
                    # print(e1, e2)
                    if rb_idx * NUM_NODES_PER_BLOCK <= e1 < (rb_idx + 1) * NUM_NODES_PER_BLOCK and e1 < e2:
                        # print(e1, e2)
                        if rb_idx * NUM_NODES_PER_BLOCK <= e2 < (rb_idx + 1) * NUM_NODES_PER_BLOCK:
                            inside.append([e1, e2])
                        else:
                            outside.append([e1, e2])
                    elif e1 < e2:
                        others.append([e1, e2])

                ### Rewiring        
                if len(inside) + len(outside) > NUM_NODES_PER_BLOCK * (NUM_NODES_PER_BLOCK - 1) / 2:
                    try:
                        for i in range(rb_idx * NUM_NODES_PER_BLOCK, (rb_idx + 1) * NUM_NODES_PER_BLOCK):
                            for j in range(i + 1, (rb_idx + 1) * NUM_NODES_PER_BLOCK):
                                if [i, j] not in inside:
                                    idx = np.random.choice(np.where(np.array(outside)[:, 0] == i)[0])
                                    outside[idx][1] = j

                        edge_index = np.concatenate([inside, outside, others], axis=0)
                        edge_index = np.concatenate([edge_index, edge_index[:, [1, 0]]], axis=0).T # to undirected 
                        edge_index = torch.Tensor(edge_index).to(torch.int64)
                    except:
                        continue
                else:
                    continue

            ### Check connectivity
            if len(np.unique(edge_index)) == NUM_NODES_PER_BLOCK * NUM_BLOCKS:
                break 

    data.edge_index = edge_index
    data.num_nodes = NUM_NODES_PER_BLOCK * NUM_BLOCKS
    return data 

def generate_sbm_task2(NUM_NODES_PER_BLOCK, NUM_BLOCKS, INTER_PROB, INTRA_PROB, clique_ratio):
    block_sizes = np.ones(NUM_BLOCKS) * NUM_NODES_PER_BLOCK
    edge_probs = np.ones((NUM_BLOCKS, NUM_BLOCKS)) * INTER_PROB
    np.fill_diagonal(edge_probs, INTRA_PROB)

    data = Data()
    data.y = torch.Tensor([clique_ratio]).to(torch.float32)        
    while True:
        edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
        block_num_edges = compute_block_num_edges(edge_index, NUM_NODES_PER_BLOCK, NUM_BLOCKS)
        has_clique = max(block_num_edges) == NUM_NODES_PER_BLOCK * (NUM_NODES_PER_BLOCK - 1)

        if not has_clique:
            for rb_idx in np.random.choice(range(NUM_BLOCKS), int(NUM_BLOCKS * clique_ratio), replace=False):
                ### Find edges inside and outside the block
                inside, outside, others = [], [], []
                for e1, e2 in edge_index.T.numpy():
                    if rb_idx * NUM_NODES_PER_BLOCK <= e1 < (rb_idx + 1) * NUM_NODES_PER_BLOCK and e1 < e2:
                        if rb_idx * NUM_NODES_PER_BLOCK <= e2 < (rb_idx + 1) * NUM_NODES_PER_BLOCK:
                            inside.append([e1, e2])
                        else:
                            outside.append([e1, e2])
                    elif e1 < e2:
                        others.append([e1, e2])

                ### Rewiring
                error = False
                if len(inside) + len(outside) > NUM_NODES_PER_BLOCK * (NUM_NODES_PER_BLOCK - 1) / 2:
                    try:
                        for i in range(rb_idx * NUM_NODES_PER_BLOCK, (rb_idx + 1) * NUM_NODES_PER_BLOCK):
                            for j in range(i + 1, (rb_idx + 1) * NUM_NODES_PER_BLOCK):
                                if [i, j] not in inside:
                                    idx = np.random.choice(np.where(np.array(outside)[:, 0] == i)[0])
                                    outside[idx][1] = j

                        edge_index = np.concatenate([inside, outside, others], axis=0)
                        edge_index = np.concatenate([edge_index, edge_index[:, [1, 0]]], axis=0).T # to undirected 
                        edge_index = torch.Tensor(edge_index).to(torch.int64)
                    except:
                        error = True
                        break
                else:
                    error = True
                    break
            if error:
                continue
        else:
            continue
            
        block_num_edges = compute_block_num_edges(edge_index, NUM_NODES_PER_BLOCK, NUM_BLOCKS)
        clique_num = len(np.where(block_num_edges == NUM_NODES_PER_BLOCK * (NUM_NODES_PER_BLOCK - 1))[0])
        if clique_num != int(NUM_BLOCKS * clique_ratio):
            continue

        ### Check connectivity
        if len(np.unique(edge_index)) == NUM_NODES_PER_BLOCK * NUM_BLOCKS:
            break 
    
    data.edge_index = edge_index
    data.num_nodes = NUM_NODES_PER_BLOCK * NUM_BLOCKS
    return data 

import networkx as nx
from networkx.algorithms.distance_measures import diameter
def generate_sbm_task3(NUM_NODES_PER_BLOCK, NUM_BLOCKS, INTRA_PROB):
    block_sizes = np.ones(NUM_BLOCKS) * NUM_NODES_PER_BLOCK
    edge_probs = np.zeros((NUM_BLOCKS, NUM_BLOCKS))
    np.fill_diagonal(edge_probs, INTRA_PROB)
    
    while True:
        data = Data()
        edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs).T
        eid = np.random.randint(0, len(edge_index), int(len(edge_index) * 0.9))
        edge_index_frm, edge_index_to = edge_index[eid].T

        ### Sample edges between blocks
        edge_comb = np.concatenate([[[i, j] for j in range(i+1, NUM_BLOCKS)] for i in range(NUM_BLOCKS-1)])
        count = 0
        while count != NUM_BLOCKS:
            rs_edge = edge_comb[np.random.choice(len(edge_comb), NUM_BLOCKS-1, replace=False)]
            count = len(np.unique(rs_edge))

        for b1, b2 in rs_edge:
            frm = np.random.randint(b1 * NUM_NODES_PER_BLOCK, (b1 + 1) * NUM_NODES_PER_BLOCK)
            to = np.random.randint(b2 * NUM_NODES_PER_BLOCK, (b2 + 1) * NUM_NODES_PER_BLOCK)
            edge_index_frm = np.concatenate([edge_index_frm, [frm, to]])
            edge_index_to = np.concatenate([edge_index_to, [to, frm]])        

        G = nx.Graph()
        G.add_edges_from(np.array([edge_index_frm, edge_index_to]).T)

        try:
            y = diameter(G)
            data.y = torch.Tensor([y]).to(torch.float32)
            break
        except:
            continue
    
    edge_index = np.unique(np.array([edge_index_frm, edge_index_to]).T, axis=1)
    edge_index = torch.Tensor(edge_index[np.argsort(edge_index[:, 0])].T)
    data.edge_index = torch.Tensor(edge_index).to(torch.long)
    data.num_nodes = NUM_NODES_PER_BLOCK * NUM_BLOCKS
    return data