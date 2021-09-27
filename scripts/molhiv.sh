python -m train.mol --config train/configs/molhiv_sampling.yaml sampling.redundancy 3

python -m train.mol --config train/configs/molhiv_sampling.yaml sampling.redundancy 2 train.dropout 0.3 model.hidden_size 80


python -m train.mol --config train/configs/molhiv_sampling.yaml model.hidden_size 80 model.embs "(0,)" sampling.redundancy 2 train.dropout 0.3 

python -m train.mol model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 
python -m train.mol model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 train.dropout 0.3

python -m train.mol --config train/configs/molhiv_sampling.yaml model.hidden_size 80 model.embs "(0,)" model.hidden_size 64 model.num_layers 2 sampling.redundancy 2