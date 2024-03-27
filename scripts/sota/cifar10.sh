# GIN
python -m train.cifar10 model.mini_layers 0 device 0 &
# GIN-AK
python -m train.cifar10 model.mini_layers 1 device 1 &

# GCN
python -m train.cifar10 model.mini_layers 0 model.gnn_type GCNConv device 2 &
# GCN-AK
python -m train.cifar10 model.mini_layers 1 model.gnn_type GCNConv device 3 &
# SimplifiedPNA
python -m train.cifar10 model.mini_layers 0 model.gnn_type SimplifiedPNAConv device 4 &
# SimplifiedPNA-AK
python -m train.cifar10 model.mini_layers 1 model.gnn_type SimplifiedPNAConv device 5 train.batch_size 40 & 

python -m train.cifar10 --config train/configs/cifar10_sampling.yaml model.mini_layers 1 model.gnn_type SimplifiedPNAConv sampling.redundancy 3 sampling.random_rate 0.2 device 4 



# # GAT
# python -m train.cifar10 model.mini_layers 0 model.gnn_type GATConv
# # GAT-AK
# python -m train.cifar10 model.mini_layers 1 model.gnn_type GATConv
# # GatedGCN
# python -m train.cifar10 model.mini_layers 0 model.gnn_type ResGatedGraphConv 
# # GatedGCM-AK
# python -m train.cifar10 model.mini_layers 1 model.gnn_type ResGatedGraphConv 