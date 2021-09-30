# GIN
python -m train.zinc model.mini_layers 0
# GIN-AK
python -m train.zinc 

# GCN
# python -m train.zinc model.mini_layers 0 model.gnn_type GCNConv
# GCN-AK
python -m train.zinc model.mini_layers 1 model.gnn_type GCNConv
# SimplifiedPNA
python -m train.zinc model.mini_layers 0 model.gnn_type SimplifiedPNAConv
# SimplifiedPNA-AK
python -m train.zinc model.mini_layers 1 model.gnn_type SimplifiedPNAConv

# GAT
python -m train.zinc model.mini_layers 0 model.gnn_type GATConv
# GAT-AK
python -m train.zinc model.mini_layers 1 model.gnn_type GATConv
# GatedGCN
python -m train.zinc model.mini_layers 0 model.gnn_type ResGatedGraphConv 
# GatedGCM-AK
python -m train.zinc model.mini_layers 1 model.gnn_type ResGatedGraphConv 


# 