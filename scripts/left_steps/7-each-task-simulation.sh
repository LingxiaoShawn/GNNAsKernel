
# counting substructure
# tasks=(0 1 2 3)
# for task in "${tasks[@]}"; do
#     # python -m train.counting task $task num_workers 8 model.mini_layers 0 model.gnn_type GCNConv         
#     # python -m train.counting task $task num_workers 8 model.mini_layers 0 model.gnn_type GINEConv        
#     python -m train.counting task $task num_workers 8 model.num_layers 4 model.mini_layers 0 model.gnn_type SimplifiedPNAConv 
# done

# # regress graph property
# tasks=(0 1 2)
# for task in "${tasks[@]}"; do
#     # python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type GCNConv        
#     # python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type GINEConv       
#     python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type SimplifiedPNAConv  
# done

# additional ablation study

tasks=(0 1 2 3)
for task in "${tasks[@]}"; do
    # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(0, 1)"  model.gnn_type GINEConv
    # python -m train.counting task $task num_workers 8 model.mini_layers 1 model.embs "(1,)"  model.gnn_type GINEConv 
    python -m train.counting task $task num_workers 8 model.mini_layers 1 model.mlp_layers 2 model.embs "(0, 1)" model.hops_dim 0 model.gnn_type GINEConv 
    python -m train.counting task $task num_workers 8 model.mini_layers 1 model.mlp_layers 2 model.hops_dim 0 model.gnn_type GINEConv         
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    # python -m train.graph_property task $task num_workers 8 model.mini_layers 1 model.embs "(0, 1)" model.gnn_type GINEConv 
    # python -m train.graph_property task $task num_workers 8 model.mini_layers 1 model.embs "(1,)" model.gnn_type GINEConv
    python -m train.graph_property task $task num_workers 8 model.mini_layers 1 model.mlp_layers 2 model.embs "(0, 1)" model.hops_dim 0 model.gnn_type GINEConv 
    python -m train.graph_property task $task num_workers 8 model.mini_layers 1 model.mlp_layers 2 model.hops_dim 0 model.gnn_type GINEConv         
done

