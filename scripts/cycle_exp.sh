# # task 1 
# python -m train.clique model.mini_layers 1 model.num_layers 6 task 1 
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 1 

# # task 2
# python -m train.clique model.mini_layers 1 model.num_layers 6 task 2 subgraph.hops 2
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 2 subgraph.hops 2

# # task 3
# python -m train.clique model.mini_layers 1 model.num_layers 6 task 3 
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 3 


# task 1 
# python -m train.clique model.mini_layers 0 model.num_layers 6 model.gnn_type PPGN task 1 train.lr 0.0002
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 1 

# task 2
# python -m train.clique model.mini_layers 0 model.num_layers 6 model.gnn_type PPGN  task 2 subgraph.hops 2
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 2 subgraph.hops 2

# task 3
python -m train.clique model.mini_layers 0 model.num_layers 6 model.gnn_type PPGN  task 3 
# python -m train.clique model.mini_layers 0 model.num_layers 6 task 3 