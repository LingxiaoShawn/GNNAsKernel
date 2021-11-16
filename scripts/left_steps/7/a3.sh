gpu=3
# gnn_type=PPGN
# mini=0
# # this is PPGN

# tasks=(0 1 2 3 4)
# for task in "${tasks[@]}"; do
#     python -m train.counting task $task num_workers 10 model.mini_layers $mini  model.gnn_type $gnn_type device $gpu
# done

# tasks=(0 1 2)
# for task in "${tasks[@]}"; do
#     python -m train.graph_property task $task num_workers 10 model.mini_layers $mini  model.gnn_type $gnn_type device $gpu
# done


tasks=(0 1 2 3)
for task in "${tasks[@]}"; do
    # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(0, 1)"  model.gnn_type GINEConv
    # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(1,)"  model.gnn_type GINEConv 
    python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 2 model.mlp_layers 1 model.embs "(0, 1)" model.hops_dim 0 model.gnn_type GINEConv device $gpu
    python -m train.counting task $task num_workers 8 model.mini_layers 3 model.num_layers 2 model.mlp_layers 1 model.hops_dim 0 model.gnn_type GINEConv                     device $gpu
done