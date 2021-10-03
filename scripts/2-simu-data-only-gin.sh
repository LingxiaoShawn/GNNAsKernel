
# for combined tasks
layers=(2 4 6 8)
for layer in "${layers[@]}"; do
    # GIN
    python -m train.graph_property model.num_layers $layer  model.mini_layers 0
    python -m train.counting model.num_layers $layer  model.mini_layers 0
    # GIN-AK 
    python -m train.graph_property model.num_layers $layer  model.mini_layers 1
    python -m train.counting model.num_layers $layer  model.mini_layers 1
    python -m train.graph_property model.num_layers $layer  model.mini_layers 2
    python -m train.counting model.num_layers $layer  model.mini_layers 2
done

layers=(2 4 6 8)
for layer in "${layers[@]}"; do
    # GIN-AK 
    python -m train.graph_property model.num_layers 1   model.mini_layers $layer
    python -m train.counting model.num_layers 1         model.mini_layers $layer
done

layers=(4 6 8)
for layer in "${layers[@]}"; do
    # GIN-AK 
    python -m train.graph_property model.num_layers 2   model.mini_layers $layer
    python -m train.counting model.num_layers  2        model.mini_layers $layer
done


# run GCN and PNA 
python -m train.counting model.gnn_type GCNConv
python -m train.counting model.gnn_type SimplifiedPNAConv
python -m train.graph_property model.gnn_type GCNConv
python -m train.graph_property model.gnn_type SimplifiedPNAConv