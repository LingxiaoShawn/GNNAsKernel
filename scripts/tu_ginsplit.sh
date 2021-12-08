batch_sizes=(32 128)
lrs=(0.01 0.001)
hiddens=(16 32 64)
dropouts=(0 0.2 0.4)
pools=(mean add)

dataset=MUTAG
########## Tuning GIN first 
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 64  train.lr 0.01 model.pool add  device  2 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 16  train.lr 0.01 model.pool mean  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  train.dropout 0.3   device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  train.batch_size 32 device 5 &
# wait

########## Tuning GIN-AK 1 layer first 
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 64  train.lr 0.01 model.pool add   device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 32  train.lr 0.001 model.pool add  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  train.dropout 0.3   device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  train.batch_size 32 device 5 &
# wait

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)" device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,)" device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(2,)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,1)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,2)" device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,2)" device 5 &
# wait

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)"  device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,1)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)"  device 3 train.dropout 0.25 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 4 train.dropout 0.25 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,1)" device 5 train.dropout 0.25 &



dataset=PTC
########## Tuning GIN first 
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 64  train.lr 0.01 model.pool mean  device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 16  train.lr 0.01 model.pool mean  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 128 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.001 model.pool mean  device 5 &

########## Tuning GIN-AK 1 layer first

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add    device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.01 model.pool mean   device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool mean  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 16  train.lr 0.01 model.pool mean   device 5 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(0,)" device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(2,)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(0,1)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(0,2)" device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool mean model.embs_combine_mode add model.embs "(1,2)" device 5 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 2 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 3 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode concat model.embs "(0,1,2)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 2 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode concat model.embs "(0,1,2)" device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 3 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode concat model.embs "(0,1,2)" device 5 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01  model.pool mean model.embs_combine_mode add model.embs "(0,)" device 0 train.lr_decay 0.8 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 1 train.lr_decay 0.8 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01  model.pool mean model.embs_combine_mode add model.embs "(1,)" device 2 train.lr_decay 0.8 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 20  train.lr 0.01  model.pool mean model.embs_combine_mode add model.embs "(0,)" device 3 train.lr_decay 0.8 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 20  train.lr 0.001 model.pool mean model.embs_combine_mode add model.embs "(1,)" device 4 train.lr_decay 0.8 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 20  train.lr 0.01  model.pool mean model.embs_combine_mode add model.embs "(1,)" device 5 train.lr_decay 0.8 &

# dataset=PROTEINS
########## Tuning GIN first 
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 64  train.lr 0.01 model.pool mean  device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 16  train.lr 0.01 model.pool mean  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 128 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool mean  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.001 model.pool mean  device 5 &

########## Tuning GIN-AK 1 layer first

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 2 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add    device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.01 model.pool add   device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.001 model.pool add  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 model.pool add  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 16  train.lr 0.01 model.pool add   device 5 &


# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)" device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 2 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)" device 1 &
# python -m train.tu_datasets_gin_split model.num_layers 3 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode concat model.embs "(0,1)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 2 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode concat model.embs "(0,1)" device 4 &
# python -m train.tu_datasets_gin_split model.num_layers 3 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.01 model.pool add model.embs_combine_mode concat model.embs "(0,1)" device 5 &


dataset=NCI1
########## Tuning GIN first 
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.001 model.pool add  device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 64  train.lr 0.001 model.pool add  device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 32  train.lr 0.001 model.pool mean  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 64  train.lr 0.001 model.pool mean  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32 dataset $dataset model.hidden_size 128  train.lr 0.001 model.pool mean  device 5 &

########## Tuning GIN-AK 1 layer first

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32   train.lr 0.001 model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64   train.lr 0.001 model.pool add   device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.001 model.pool add   device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64   train.lr 0.001 model.pool add model.embs_combine_mode concat model.embs "(0,1)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64   train.lr 0.001 model.pool add model.embs_combine_mode concat model.embs "(0,2)" device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64   train.lr 0.001 model.pool add model.embs_combine_mode concat model.embs "(1,2)" device 5 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add train.dropout 0.25                             device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add model.embs_combine_mode add                    device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(0,1)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(1,2)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 64  train.lr 0.001 model.pool add model.embs_combine_mode concat model.embs "(0,1)" train.dropout 0.25 device 5 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 96   train.lr 0.001 model.pool add model.embs_combine_mode add                     device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.001 model.pool add model.embs_combine_mode add                     device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.001 model.pool add model.embs_combine_mode add train.dropout 0.2   device 2 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 96   train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.001 model.pool add model.embs_combine_mode add model.embs "(1,)"  device 5 train.dropout 0.2 &

# wait 

dataset=IMDBBINARY

########## Tuning GIN first 
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32  dataset $dataset model.hidden_size 32  train.lr 0.001  model.pool add   device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32  dataset $dataset model.hidden_size 32  train.lr 0.001  model.pool mean  device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32  dataset $dataset model.hidden_size 64  train.lr 0.001  model.pool add  device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32  dataset $dataset model.hidden_size 16  train.lr 0.001  model.pool add  device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 128 dataset $dataset model.hidden_size 32  train.lr 0.001  model.pool add  device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 0 train.batch_size 32  dataset $dataset model.hidden_size 32  train.lr 0.01 model.pool add  device 5 &

# wait 

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,)" device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,)" device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(2,)" device 2 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,1)" device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(0,2)" device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add model.embs "(1,2)" device 5 &

# wait 

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 16  model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add device 0 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 64  model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add device 1 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 128 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode add device 2 &

# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 16  model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode concat device 3 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 64  model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode concat device 4 &
# python -m train.tu_datasets_gin_split model.mini_layers 1 dataset $dataset train.batch_size 128 model.hidden_size 32  train.lr 0.01 model.pool add model.embs_combine_mode concat device 5 &


# Next: study hops 

# dataset=IMDBBINARY
# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 6  subgraph.walk_repeat 4 device 0 &
# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 8  subgraph.walk_repeat 3 device 1 &
# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 10 subgraph.walk_repeat 3 device 2 &

# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 5 subgraph.walk_repeat 4 device 3 train.dropout 0.25&
# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 7 subgraph.walk_repeat 3 device 4 train.dropout 0.25&
# python -m train.tu_datasets_gin_split num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 4 subgraph.walk_repeat 5 device 5 train.dropout 0.25&



dataset=REDDIT-BINARY
python -m train.tu_datasets num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 6  subgraph.walk_repeat 5 device 0 &
python -m train.tu_datasets num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 8  subgraph.walk_repeat 4 device 1 &
python -m train.tu_datasets num_workers 12 model.mini_layers 1 dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 10 subgraph.walk_repeat 3 device 2 &

python -m train.tu_datasets num_workers 12 model.mini_layers 1 model.pool mean dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 6  subgraph.walk_repeat 5 device 3 &
python -m train.tu_datasets num_workers 12 model.mini_layers 1 model.pool mean dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 8  subgraph.walk_repeat 4 device 4 &
python -m train.tu_datasets num_workers 12 model.mini_layers 1 model.pool mean dataset $dataset train.batch_size 32 model.hidden_size 128  train.lr 0.01 subgraph.hops 0 subgraph.walk_length 10 subgraph.walk_repeat 3 device 5 &




