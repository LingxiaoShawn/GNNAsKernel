# # zinc
# python -m train.zinc model.embs "(0,1)"  device 4
# python -m train.zinc model.embs "(0,2)"  device 4
# python -m train.zinc model.embs "(1,2)"  device 4


# # cifar10
# python -m train.cifar10 model.embs "(0,1)"  device 4
# python -m train.cifar10 model.embs "(0,2)"  num_workers 12 device 4 &
# python -m train.cifar10 model.embs "(1,2)"  num_workers 12 device 5 &
# python -m train.cifar10 subgraph.hops  1 num_workers 12 device 2 &
# python -m train.cifar10 model.hops_dim 0 num_workers 12 device 3 &

## additional ablation study
# python -m train.cifar10 model.embs "(0,)"  device 2 &
# python -m train.cifar10 model.embs "(1,)"  device 3 &
# python -m train.cifar10 model.embs "(2,)"  device 4 &
# python -m train.cifar10 model.embs "(1,)" model.hops_dim 0 device 5 &

# python -m train.zinc model.embs "(0,)"  device 1
# python -m train.zinc model.embs "(1,)"  device 1
# python -m train.zinc model.embs "(2,)"  device 1
# python -m train.zinc model.embs "(1,)" model.hops_dim 0 device 1



python -m train.cifar10 model.embs "(0,1)" model.hops_dim 0 device 5 &
python -m train.zinc model.embs "(0,1)" model.hops_dim 0 device 1 &
