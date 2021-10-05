gpu=0
gnn_type=GCNConv
# this is gnn-ak

tasks=(0 1 2 3 4)
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 10 model.gnn_type $gnn_type device $gpu
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 10 model.gnn_type $gnn_type device $gpu
done