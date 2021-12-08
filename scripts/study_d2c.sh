# tasks=(0 1 2 3)
# for task in "${tasks[@]}"; do
#     # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(0, 1)"  model.gnn_type GINEConv
#     # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(1,)"  model.gnn_type GINEConv 
#     python -m train.counting task $task num_workers 8 model.mini_layers 2 model.num_layers 3 model.mlp_layers 1 model.embs "(0, 1)" model.hops_dim 0 model.gnn_type GINEConv device $gpu
#     python -m train.counting task $task num_workers 8 model.mini_layers 2 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv                     device $gpu
# done

task=0

# python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 1 &
# python -m train.counting task $task num_workers 8 model.mini_layers 4 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 2 &
# python -m train.counting task $task num_workers 8 model.mini_layers 6 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 3 &


# python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 4 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 1 &
# python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 5 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 2 &
# python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 6 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 3 &

# python -m train.graph_property task $task num_workers 8 model.mini_layers 3 model.num_layers 4 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 1 &
# python -m train.graph_property task $task num_workers 8 model.mini_layers 3 model.num_layers 5 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 2 &
# python -m train.graph_property task $task num_workers 8 model.mini_layers 3 model.num_layers 6 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 3 &

# python -m train.graph_property task $task num_workers 8 model.mini_layers 3 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 1 &
# python -m train.graph_property task $task num_workers 8 model.mini_layers 4 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 2 &
# python -m train.graph_property task $task num_workers 8 model.mini_layers 6 model.num_layers 3 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv train.runs 1 device 3 &


tasks=(0 1 2 3)
gpu=0
a=3
b=2
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 8 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv  model.embs "(0, 1)"  device $gpu
    # python -m train.counting task $task num_workers 8 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 8 model.mini_layers $a model.num_layers $b model.hops_dim 0 model.gnn_type GINEConv  model.embs "(0, 1)"   device $gpu
    # python -m train.graph_property task $task num_workers 8 model.mini_layers $a model.num_layers $b model.hops_dim 16 model.gnn_type GINEConv   device $gpu      
done