
# 1. zinc, GCN-AK GIN-AK PNA-AK
# python -m train.zinc --config train/configs/molhiv_sampling.yaml
python -m train.zinc --config train/configs/molhiv_sampling.yaml model.gnn_type GCNConv
python -m train.zinc --config train/configs/molhiv_sampling.yaml model.gnn_type SimplifiedPNAConv

# # 2. molhiv
# python -m train.molhiv --config train/configs/molhiv_sampling.yaml
# python -m train.molhiv --config train/configs/molhiv_sampling.yaml model.gnn_type GCNConv
# python -m train.molhiv --config train/configs/molhiv_sampling.yaml model.gnn_type SimplifiedPNAConv

# # 3. molpcba 
# python -m train.molpcba --config train/configs/molhiv_sampling.yaml
# python -m train.molpcba --config train/configs/molhiv_sampling.yaml model.gnn_type GCNConv
# python -m train.molpcba --config train/configs/molhiv_sampling.yaml model.gnn_type SimplifiedPNAConv

# # 4. cifar10
# python -m train.cifar10 --config train/configs/cifar10_sampling.yaml
# python -m train.cifar10 --config train/configs/cifar10_sampling.yaml model.gnn_type GCNConv
# python -m train.cifar10 --config train/configs/cifar10_sampling.yaml model.gnn_type SimplifiedPNAConv

# # 5. pattern
# python -m train.pattern --config train/configs/pattern_sampling.yaml
# python -m train.pattern --config train/configs/pattern_sampling.yaml model.gnn_type GCNConv
# python -m train.pattern --config train/configs/pattern_sampling.yaml model.gnn_type SimplifiedPNAConv
