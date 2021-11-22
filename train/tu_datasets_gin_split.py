import torch
from core.config import cfg, update_cfg
from core.train_helper import run_k_fold 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import TUDatasetGINSplit
from torch_geometric.transforms import Constant, Compose
from core.data import calculate_stats
import shutil

class ToLong(object):
    def __call__(self, data):
        data.x = data.x.long()
        return data

def create_dataset(cfg):
    torch.set_num_threads(cfg.num_workers)
    dataset = TUDatasetGINSplit(cfg.dataset)

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
    if dataset.data.x is None:
        transform = Compose([Constant(value=1), ToLong(), transform])
        transform_eval = Compose([Constant(value=1), ToLong(), transform_eval])

    cfg.n_in = None if dataset.data.x is None else dataset.data.x.size(-1) # continous feature when existing
    cfg.n_in_edge = None if dataset.data.edge_attr is None else dataset.data.edge_attr.size(-1) # continous feature when existing
    cfg.n_out = dataset.num_classes
    
    print('------------Stats--------------')
    calculate_stats(dataset)
    return dataset, transform, transform_eval

def create_model(cfg):
    model = GNNAsKernel(cfg.n_in, cfg.n_in_edge, 
                        nhid=cfg.model.hidden_size, 
                        nout=cfg.n_out, 
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
    return model

def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(data.y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/tu.yaml')
    cfg = update_cfg(cfg)
    run_k_fold(cfg, create_dataset, create_model, train, test)