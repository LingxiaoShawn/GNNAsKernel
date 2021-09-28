
import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

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

    dataset = PygGraphPropPredDataset(cfg.dataset, 'data')
    split_idx = dataset.get_idx_split()
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_dataset.transform, val_dataset.transform, test_dataset.transform = transform, transform_eval, transform_eval

    # When without randomness, transform the data to save a bit time
    torch.set_num_threads(cfg.num_workers)
    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=128 if cfg.dataset=='ogbg-molpcba' else 1, 
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
    criterion = torch.nn.BCEWithLogitsLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        out = model(data)[mask]
        y = data.y.to(torch.float)[mask]
        loss = criterion(out, y)
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
        y_preds.append(model(data))
        y_trues.append(data.y)

    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/molhiv.yaml')
    cfg = update_cfg(cfg)
    evaluator = Evaluator(cfg.dataset)
    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator)   