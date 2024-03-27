import torch
from core.config import cfg, update_cfg
from core.transform_utils.EVD_transform import EVDTransform
from core.transform_utils.combine_transforms import CombineTransforms
from core.transform import SubgraphsTransform
from torch_geometric.datasets import ZINC
from core.train_helper import run 

from core.model_enhanced import GNNAsKernelEnhanced

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    pre_transform = EVDTransform()
    transform_1 = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True,
                                   FGSD=False,
                                   FGSD_k=cfg.subgraph.FGSD_k,
                                   FGSD_p=cfg.subgraph.FGSD_p,
                                   FGSD_q=cfg.subgraph.FGSD_q)
    transform_2 = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True,
                                   FGSD=True,
                                   FGSD_k=cfg.subgraph.FGSD_k,
                                   FGSD_p=cfg.subgraph.FGSD_p,
                                   FGSD_q=cfg.subgraph.FGSD_q)
                                   
    combined_transform = CombineTransforms([transform_1, transform_2])
    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', pre_transform=pre_transform, transform=combined_transform)
    val_dataset = ZINC(root, subset=True, split='val', pre_transform=pre_transform, transform=combined_transform) 
    test_dataset = ZINC(root, subset=True, split='test', pre_transform=pre_transform, transform=combined_transform)   

    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernelEnhanced(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=1, 
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
                        online=cfg.subgraph.online,
                        num_transforms=2) 
    return model


def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0 
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = (model(data).squeeze() - y).abs().mean()
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    # get config 
    cfg.merge_from_file('train/configs/zinc.yaml')
    cfg = update_cfg(cfg)
    dataset = create_dataset(cfg)
    loader =  DataLoader(dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    run(cfg, create_dataset, create_model, train, test)
