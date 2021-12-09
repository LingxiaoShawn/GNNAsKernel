import torch
from torch_geometric import transforms
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import CliqueSBM, calculate_stats

from torch_geometric.transforms import Compose, BaseTransform
class Constant(BaseTransform):
    def __call__(self, data):
        for store in data.node_stores:
            c = torch.full((store.num_nodes, 1), 1, dtype=torch.int64)
            store.x = c
        return data

def create_dataset(cfg):
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

    root = 'data/clique_sbm'
    transform = Compose([Constant(), transform])
    transform_eval = Compose([Constant(), transform_eval])
    train_dataset = CliqueSBM(root, split='train', task=cfg.task, transform=transform)
    val_dataset = CliqueSBM(root, split='val', task=cfg.task, transform=transform_eval)
    test_dataset = CliqueSBM(root, split='test', task=cfg.task, transform=transform_eval)


    # When without randomness, transform the data to save a bit time
    # torch.set_num_threads(cfg.num_workers)
    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 
    
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=2 if cfg.task == 1 else 1, 
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
    criterion = torch.nn.CrossEntropyLoss() if model.task == 1 else torch.nn.L1Loss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    if model.task == 1:
        y_preds, y_trues = [], []
        for data in loader:
            data = data.to(device)
            y_preds.append(torch.argmax(model(data), dim=-1))
            y_trues.append(data.y)
        y_preds = torch.cat(y_preds, -1)
        y_trues = torch.cat(y_trues, -1)
        perf = (y_preds == y_trues).float().mean()
    else:
        total_error = 0
        N = 0
        for data in loader:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
            total_error += (model(data).squeeze() - y).abs().sum().item()
            N += num_graphs
        perf = - total_error / N
    return perf



if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/clique.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)   