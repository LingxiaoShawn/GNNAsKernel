python -m train.molhiv train.dropout 0.5 model.num_layers 2 model.mini_layers 0  num_workers 12 train.runs 5 device 0 &
python -m train.molhiv train.dropout 0.5 model.num_layers 6 model.mini_layers 0  num_workers 12 train.runs 5 device 1 &
python -m train.molhiv train.dropout 0.5 model.num_layers 2 model.embs "(2,)"    num_workers 12 train.runs 5 device 2 &
python -m train.molhiv train.dropout 0.5 model.num_layers 2 model.embs "(2,0)"   num_workers 12 train.runs 5 device 3 &
python -m train.molhiv train.dropout 0.5 model.num_layers 6 model.embs "(2,)"    num_workers 12 train.runs 5 device 4 &
python -m train.molhiv train.dropout 0.5 model.num_layers 6 model.embs "(2,0)"   num_workers 12 train.runs 5 device 5 &

