import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform
from core.data import calculate_stats

from torch_geometric.datasets import GNNBenchmarkDataset
from sklearn.metrics import confusion_matrix
import numpy as np

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
    root = 'data'
    train_dataset = GNNBenchmarkDataset(root, 'PATTERN', split='train', transform=transform)
    val_dataset = GNNBenchmarkDataset(root, 'PATTERN', split='val', transform=transform_eval)
    test_dataset = GNNBenchmarkDataset(root, 'PATTERN', split='test', transform=transform_eval)

    train_dataset.data.x = train_dataset.data.x.long()
    val_dataset.data.x = val_dataset.data.x.long()
    test_dataset.data.x = test_dataset.data.x.long()

    # When without randomness, transform the data to save a bit time
    torch.set_num_threads(cfg.num_workers)
    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 
    print('------------Train--------------')
    calculate_stats(train_dataset)
    print('------------Validation--------------')
    calculate_stats(val_dataset)
    print('------------Test--------------')
    calculate_stats(test_dataset)
    print('------------------------------')
    # exit(0)
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=2, 
                        node_embedding=True,
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
    train_acc = 0
    N = 0 
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # print(data.x.shape, out.shape, data.y.shape)
        # exit(0)
        loss = loss_func(out, data.y)
        loss.backward()
        total_loss += loss.item() 
        train_acc += accuracy_SBM(out, data.y) * data.num_graphs
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
    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    return accuracy_SBM(y_preds, y_trues)   

def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc

def loss_func(pred, label):
    # calculating label weights for weighted loss computation
    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    # cluster_sizes = torch.zeros(pred.size(-1)).long().to(label.device)
    cluster_sizes = label.new_zeros(pred.size(-1))
    cluster_sizes[torch.unique(label)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes>0).float()
    
    # weighted cross-entropy for unbalanced classes
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, label)
    return loss


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/pattern.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)