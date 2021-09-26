python -m train.zinc model.embs "(1,2)"    model.embs_combine_mode 'add'    device 1 &
python -m train.zinc model.embs "(1,2)"    model.embs_combine_mode 'concat' device 2 &
python -m train.zinc model.embs "(1,)"     model.embs_combine_mode 'add'    device 3 &
python -m train.zinc model.embs "(2,)"     model.embs_combine_mode 'add'    device 4 &
python -m train.zinc model.embs "(0,1,2)"  model.embs_combine_mode 'add'    device 5 &

wait

python -m train.zinc model.hops_dim 0          device 0 &
python -m train.zinc model.virtual_node   True device 1 &
python -m train.zinc model.use_normal_gnn True device 2 &
python -m train.zinc model.hidden_size    256  device 3 &
python -m train.zinc model.num_layers  8       device 4 &
python -m train.zinc model.hidden_size    256  train.dropout 0.25 device 5 &

wait 

python -m train.zinc sampling.mode 'shortest_path' sampling.stride 5 sampling.redundancy 1 device 0 & 
python -m train.zinc sampling.mode 'shortest_path' sampling.stride 5 sampling.redundancy 2 device 1 & 
python -m train.zinc sampling.mode 'shortest_path' sampling.stride 5 sampling.redundancy 3 device 2 & 
python -m train.zinc sampling.mode 'shortest_path' sampling.stride 5 sampling.redundancy 4 device 3 & 
python -m train.zinc sampling.mode 'shortest_path' sampling.stride 5 sampling.redundancy 5 device 4 & 


wait 

python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'concat' model.hidden_size  256 train.dropout 0.25 device 0 &
python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'add'    model.hidden_size  256 device 1 & 
python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'concat' model.hidden_size  512 device 2 &
python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'add'    model.hidden_size  512 device 3 & 
python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'concat' model.hidden_size  256 model.mlp_layers 2 device 4 &
python -m train.zinc model.embs "(1,2)"  model.embs_combine_mode 'concat' model.hidden_size  256 model.mlp_layers 2 train.dropout 0.25 device 5 & 