import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import GraphPropertyDataset, calculate_stats
import numpy as np

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    # No need to do offline transformation
    transform = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True)

    transform_eval = SubgraphsTransform(cfg.subgraph.hops, 
                                        walk_length=cfg.subgraph.walk_length, 
                                        p=cfg.subgraph.walk_p, 
                                        q=cfg.subgraph.walk_q, 
                                        repeat=cfg.subgraph.walk_repeat,
                                        sampling_mode=None, 
                                        random_init=False)
    root = 'data/graphproperty'
    train_dataset = GraphPropertyDataset(root, split='train', transform=transform)
    val_dataset = GraphPropertyDataset(root, split='val', transform=transform_eval)
    test_dataset = GraphPropertyDataset(root, split='test', transform=transform_eval)

    # When without randomness, transform the data to save a bit time
    # torch.set_num_threads(cfg.num_workers)
    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 
    
    # print('------------Train--------------')
    # calculate_stats(train_dataset)
    # print('------------Validation--------------')
    # calculate_stats(val_dataset)
    # print('------------Test--------------')
    # calculate_stats(test_dataset)
    # exit(0)

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(2, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=3 if cfg.task < 0 else 1, 
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=[cfg.model.gnn_type], 
                        hop_dim=cfg.model.hops_dim,
                        use_normal_gnn=cfg.model.use_normal_gnn, 
                        vn=cfg.model.virtual_node, 
                        pooling=cfg.model.pool,
                        embs=cfg.model.embs,
                        embs_combine_mode=cfg.model.embs_combine_mode,
                        mlp_layers=cfg.model.mlp_layers,
                        dropout=cfg.train.dropout, 
                        subsampling=True if cfg.sampling.mode is not None else False,
                        online=cfg.subgraph.online) 
    model.task = cfg.task
    return model

def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    ntask = model.task
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if ntask >= 0:
            loss = (model(data).squeeze() - data.y[:,ntask:ntask+1].squeeze()).square().mean()
        else:
            loss = (model(data) - data.y).square().mean() 

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    return np.log10(total_loss / N)

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0 
    ntask = model.task
    for data in loader:
        data = data.to(device)
        if ntask >= 0:
            total_error += (model(data).squeeze() - data.y[:,ntask:ntask+1].squeeze()).square().sum().item()
        else:
            total_error += (model(data) - data.y).square().sum().item()
        N += data.num_graphs

    return - np.log10(total_error / N)

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/graph_property.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)   