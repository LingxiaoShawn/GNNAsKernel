from yacs.config import CfgNode as CN

def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ZINC'
    # Additional num of worker for data loading
    cfg.num_workers = 8
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0 
    # Additional string add to logging 
    cfg.handtune = ''
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Whether downsampling the dataset, used for large dataset for faster tuning
    cfg.downsample = False 
    # version 
    cfg.version = 'final'
    # task, for simulation datasets
    cfg.task = -1

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    # Total graph mini-batch size
    cfg.train.batch_size = 128
    # Maximal number of epochs
    cfg.train.epochs = 100
    # Number of runs with random init 
    cfg.train.runs = 3
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 50
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.train.wd = 0.
    # Dropout rate
    cfg.train.dropout = 0.
    
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GNN type used, see core.model_utils.pyg_gnn_wrapper for all options
    cfg.model.gnn_type = 'GINEConv' # change to list later
    # Hidden size of the model
    cfg.model.hidden_size = 128
    # Number of gnn layers (doesn't include #MLPs)
    cfg.model.num_layers = 4
    # Number of inner layers, 0 means use normal GNN
    cfg.model.mini_layers = 0
    # Pooling type for generaating graph/subgraph embedding from node embeddings 
    cfg.model.pool = 'add'
    # Use residual connection
    cfg.model.residual = True
    # Whether use virtual node in message passing, keep it always false
    cfg.model.virtual_node = False
    # Distance to centroid embedding dim, 0 means do not use it
    cfg.model.hops_dim = 16
    # What embedding to include
    cfg.model.embs = (0, 1, 2) # 0-Centroid, 1-Subgraph,  2-Context 
    # How to combine embs together when have more than 1 embs, choose from ['concat', 'add']
    cfg.model.embs_combine_mode = 'concat'
    # Number of MLP layers in generating subgraph embeddings
    cfg.model.mlp_layers = 1
    # Whether combine with normal gnn, only used for random walk based subgraph which doesn't cover whole 1st hop
    cfg.model.use_normal_gnn = False


    # ------------------------------------------------------------------------ #
    # Subgraph patterns options
    # ------------------------------------------------------------------------ #
    cfg.subgraph = CN()
    ##### Option 1: Egonet based subgraph
    cfg.subgraph.hops = 3
    ##### Option 2: Random walk based subgraph
    # Random walk based subgraph walk length, when 0 use egonet
    cfg.subgraph.walk_length = 0
    # We use node2vec based random walk, p is return parameter. Small p => local walk
    cfg.subgraph.walk_p = 1.0
    # q is in-out parameter. Large q => local walk
    cfg.subgraph.walk_q = 1.0
    # Number of times of repeating the random walk to reduce randomness
    cfg.subgraph.walk_repeat = 5

    # extract subgraph online or offline. Online cost more, but can help randomness
    cfg.subgraph.online = True # When use sampling subgraphs, keep it True. 

    # ------------------------------------------------------------------------ #
    # Subgraph sampling options, to improve scalability
    # ------------------------------------------------------------------------ #
    cfg.sampling = CN()
    # mode to use, choose from [shortest_path, min_set, random, None]
    cfg.sampling.mode = None # None means don't use sampling 
    # The minimum times each node is covered by all sampled subgraphs
    cfg.sampling.redundancy = 0
    # Options for shortest_path_distance based farthest sampling
    cfg.sampling.stride = 2 
    # Options for random sampling
    cfg.sampling.random_rate = 0.5
    # Reduce evaluation batch size by a factor to avoid use more meory, as test is always in full-mode
    cfg.sampling.batch_factor = 1

    return cfg
    
import os 
import argparse
# Principle means that if an option is defined in a YACS config object, 
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining, 
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.

def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, 
                         help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg 
    cfg = cfg.clone()
    
    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line 
    cfg.merge_from_list(args.opts)
       
    return cfg

"""
    Global variable
"""
cfg = set_cfg(CN())