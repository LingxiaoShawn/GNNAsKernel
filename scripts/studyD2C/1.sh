tasks=(0 1 2 3)
gpu=1
a=2
b=3
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv    device $gpu
    python -m train.counting task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv    device $gpu
    python -m train.graph_property task $task num_workers 12 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu      
done