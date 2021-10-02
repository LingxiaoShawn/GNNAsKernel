
# GCN
python -m train.pattern model.gnn_type GCNConv model.mini_layers 0 subgraph.online False device 0 & 
# PNA
python -m train.pattern model.gnn_type SimplifiedPNAConv model.mini_layers 0 subgraph.online False device 1 & 
# GCN-AK
python -m train.pattern model.gnn_type GCNConv device 2 & 
# GIN-AK
python -m train.pattern device 3 & 
# PNA-AK
python -m train.pattern model.gnn_type SimplifiedPNAConv model.num_layers 4  device 4 & 