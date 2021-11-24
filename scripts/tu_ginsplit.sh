batch_sizes=(32 128)
lrs=(0.01 0.001)
hiddens=(16 32 64)
dropouts=(0 0.2 0.4)
pools=(mean add)

# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01   device 0&
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.001  device 1&
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 64 model.embs_combine_mode 'add'    train.lr 0.01   device 2&
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 64 model.embs_combine_mode 'add'    train.lr 0.001  device 3&
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'concat' train.lr 0.01   device 4&
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'concat' train.lr 0.001  device 5&


# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 16 model.embs_combine_mode 'add'    train.lr 0.01   device 0  model.embs "(0,1)" &
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 16 model.embs_combine_mode 'add'    train.lr 0.01   device 1  model.embs "(0,2)" &
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 16 model.embs_combine_mode 'add'    train.lr 0.01   device 2  model.embs "(0,1,2)" &
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01   device 3  model.embs "(0,1)" &
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01   device 4  model.embs "(0,2)" &
# python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01   device 5  model.embs "(0,1,2)" &

python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 64 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-5  device 0  model.mini_layers 1  subgraph.hops 3 &
python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 64 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-4  device 1  model.mini_layers 1  subgraph.hops 3 &
python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 128 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-5  device 2  model.mini_layers 1  subgraph.hops 3 &
python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 128 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-4  device 3  model.mini_layers 1  subgraph.hops 3 &
python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-5  device 4  model.mini_layers 1  subgraph.hops 3 &
python -m train.tu_datasets_gin_split dataset MUTAG model.hidden_size 32 model.embs_combine_mode 'add'    train.lr 0.01  train.wd 1e-4  device 5  model.mini_layers 1  subgraph.hops 3 &



# python -m train.tu_dataset_gin_split dataset PTG train.dropout $dropout device 0