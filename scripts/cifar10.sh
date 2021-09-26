# CIFAR10 better needs 2-Layer output decoder for GNN.

# python -m train.cifar10 model.embs "(2,)"     model.embs_combine_mode 'add'                        device 1 &
# python -m train.cifar10 model.embs "(1,)"     model.embs_combine_mode 'add'                        device 2 &
# python -m train.cifar10 model.embs "(1,2)"    model.embs_combine_mode 'add'                        device 3 &
# python -m train.cifar10 model.embs "(1,2)"    model.embs_combine_mode 'concat'                     device 4 &
# python -m train.cifar10 model.embs "(1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 2  device 5 &

python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 1  device 0 &
python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 2  device 1 &
python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 1 model.num_layers 5 device 2 &
python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 1 train.dropout 0.2 device 3 &
python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 1 model.pool mean device 4 &

python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 1 device 5 handtune GNNOutMLPLayer2 &


# python -m train.cifar10 model.embs "(1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 2  device 5 &
python -m train.cifar10 model.embs "(0,1,2)"    model.embs_combine_mode 'concat' model.mlp_layers 2 device 2 handtune GNNOutMLPLayer2 &