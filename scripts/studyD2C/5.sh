tasks=(0 1 2 3)
gpu=5
a=6
b=1
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv    device $gpu model.embs "(0,1)"
    python -m train.counting task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu model.embs "(0,1)"
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv    device $gpu model.embs "(0,1)"
    python -m train.graph_property task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu model.embs "(0,1)"     
done