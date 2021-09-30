# GIN
python -m train.molpcba model.mini_layers 0
# GIN-AK
python -m train.molpcba model.mini_layers 1
# GCN
python -m train.molpcba model.mini_layers 0 model.gnn_type GCNConv
# GCN-AK
python -m train.molpcba model.mini_layers 1 model.gnn_type GCNConv
# SimplifiedPNA
python -m train.molpcba model.mini_layers 0 model.gnn_type SimplifiedPNAConv
# SimplifiedPNA-AK
python -m train.molpcba model.mini_layers 1 model.gnn_type SimplifiedPNAConv


# GIN
python -m train.molpcba train.dropout 0.3 model.mini_layers 0
# GIN-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1
# GCN
python -m train.molpcba train.dropout 0.3 model.mini_layers 0 model.gnn_type GCNConv
# GCN-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1 model.gnn_type GCNConv
# SimplifiedPNA
python -m train.molpcba train.dropout 0.3 model.mini_layers 0 model.gnn_type SimplifiedPNAConv
# SimplifiedPNA-AK
python -m train.molpcba train.dropout 0.3 model.mini_layers 1 model.gnn_type SimplifiedPNAConv