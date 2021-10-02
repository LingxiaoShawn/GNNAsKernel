# python -m train.mol --config train/configs/molhiv_sampling.yaml sampling.redundancy 3

# python -m train.mol --config train/configs/molhiv_sampling.yaml sampling.redundancy 2 train.dropout 0.3 model.hidden_size 80


# python -m train.mol --config train/configs/molhiv_sampling.yaml model.hidden_size 80 model.embs "(0,)" sampling.redundancy 2 train.dropout 0.3 

# python -m train.mol model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 
# python -m train.mol model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 train.dropout 0.3

# python -m train.mol --config train/configs/molhiv_sampling.yaml model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 sampling.redundancy 2




# python -m train.molhiv model.num_layers 2 device 0 &
# python -m train.molhiv model.num_layers 2 model.embs "(1,)" device 1 &
# python -m train.molhiv model.num_layers 2 model.embs "(2,)" device 2 &
# python -m train.molhiv model.embs "(1,)" device 3 & 
# python -m train.molhiv model.embs "(2,)" device 4 & 
# python -m train.molhiv model.embs "(0,1,2)" device 5 & 


# python -m train.molhiv model.embs "(0,)"  train.dropout 0.3 device 0 &
# python -m train.molhiv model.embs "(0,)"  train.dropout 0.5 device 1 &
# python -m train.molhiv model.embs "(2,)"  train.dropout 0.3 device 2 &
# python -m train.molhiv model.embs "(2,)"  train.dropout 0.5 device 3 &
# python -m train.molhiv model.embs "(2,0)" train.dropout 0.3 device 4 & 
# python -m train.molhiv model.embs "(0,2)" train.dropout 0.5 device 5 & 


python -m train.molhiv model.embs "(2,)" model.num_layers 4 train.dropout 0. device 0 &
python -m train.molhiv model.embs "(2,)" model.num_layers 4 train.dropout 0.3 device 1 &
python -m train.molhiv model.embs "(2,)" model.num_layers 2 train.dropout 0.3 device 2 &
python -m train.molhiv model.embs "(0,2)" model.num_layers 4 train.dropout 0. device 3 & 
python -m train.molhiv model.embs "(0,2)" model.num_layers 4 train.dropout 0.3 device 4 & 
python -m train.molhiv model.embs "(0,2)" model.num_layers 2 train.dropout 0.3 device 5 & 