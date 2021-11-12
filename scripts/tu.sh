python -m train.tu_datasets dataset PTC_MR  model.mini_layers 1  device 0 &
python -m train.tu_datasets dataset PROTEINS  model.mini_layers 1  device 1 &
python -m train.tu_datasets dataset NCI1  model.mini_layers 1  device 2 &
python -m train.tu_datasets dataset IMDB-BINARY  model.mini_layers 1  device 3 &
python -m train.tu_datasets dataset IMDB-MULTI  model.mini_layers 1  device 4 &
python -m train.tu_datasets dataset REDDIT-BINARY  model.mini_layers 1 subgraph.walk_length 8 subgraph.walk_repeat 4 device 5 &


