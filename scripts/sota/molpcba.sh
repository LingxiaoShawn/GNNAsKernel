# # GIN
# python -m train.molpcba model.mini_layers 0
# # GCN
# python -m train.molpcba model.mini_layers 0 model.gnn_type GCNConv
# # SimplifiedPNA
# python -m train.molpcba model.mini_layers 0 model.gnn_type SimplifiedPNAConv
# # GIN-AK
# python -m train.molpcba model.mini_layers 1
# # GCN-AK
# python -m train.molpcba model.mini_layers 1 model.gnn_type GCNConv
# # SimplifiedPNA-AK
# python -m train.molpcba model.mini_layers 1 model.gnn_type SimplifiedPNAConv



# # GIN
# python -m train.molpcba train.dropout 0.2 model.mini_layers 0                                     device 0 &
# # GCN
# python -m train.molpcba train.dropout 0.2 model.mini_layers 0 model.gnn_type GCNConv              device 1 &

# # SimplifiedPNA
# python -m train.molpcba train.dropout 0.2 model.mini_layers 0 model.gnn_type SimplifiedPNAConv    device 2 &

# # GCN-AK
# python -m train.molpcba train.dropout 0.2 model.mini_layers 1 model.gnn_type GCNConv              device 3 & 
# # GIN-AK
# python -m train.molpcba train.dropout 0.2 model.mini_layers 1                                     device 4 &
# # SimplifiedPNA-AK
# python -m train.molpcba train.dropout 0.2 model.mini_layers 1 model.gnn_type SimplifiedPNAConv    device 5 & 

# GCN-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1 model.gnn_type GCNConv              device 0 & 
# GIN-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1                                     device 1 &
# SimplifiedPNA-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1 model.gnn_type SimplifiedPNAConv    device 2 & 

# Sampling 

# python -m train.molpcba --config train/configs/molpcba_sampling.yaml train.dropout 0.2 model.mini_layers 1 device 0