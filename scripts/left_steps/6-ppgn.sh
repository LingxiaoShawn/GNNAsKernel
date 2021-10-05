## sr 25
hops=(1 2)
inner_layers=(1 2 3)
outer_layers=(1 2 3)

for hops in "${hops[@]}"; do
for outer in "${outer_layers[@]}"; do
for inner_layers in "${inner_layers[@]}"; do
    python -m train.sr25 subgraph.hops $hop model.mini_layers $inner model.num_layers $outer  model.gnn_type PPGN
done
done
done


## zinc 
# python -m train.zinc --config train/configs/molhiv_sampling.yaml sampling.redundancy 2  model.gnn_type PPGN model.num_layers 3  model.mini_layers 2  num_workers 14 device 5