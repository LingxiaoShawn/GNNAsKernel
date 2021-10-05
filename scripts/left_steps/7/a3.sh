gpu=3
gnn_type=PPGN
mini=0
# this is PPGN

tasks=(0 1 2 3 4)
for task in "${tasks[@]}"; do
    python -m train.counting task $task num_workers 10 model.mini_layers $mini  model.gnn_type $gnn_type device $gpu
done

tasks=(0 1 2)
for task in "${tasks[@]}"; do
    python -m train.graph_property task $task num_workers 10 model.mini_layers $mini  model.gnn_type $gnn_type device $gpu
done