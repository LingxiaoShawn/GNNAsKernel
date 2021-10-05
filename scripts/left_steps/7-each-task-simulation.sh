
# counting substructure
tasks=(0 1 2 3 4)
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 8 model.mini_layers 0 model.gnn_type GCNConv         
    python -m train.counting task $task num_workers 8 model.mini_layers 0 model.gnn_type GINEConv        
    python -m train.counting task $task num_workers 8 model.mini_layers 0 model.gnn_type SimplifiedPNAConv   
done

# regress graph property
tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type GCNConv        
    python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type GINEConv       
    python -m train.graph_property task $task num_workers 8 model.mini_layers 0 model.gnn_type SimplifiedPNA  
done
