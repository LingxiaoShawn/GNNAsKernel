# redundancys=(1 2 3 4 5)
# # 1. zinc, GCN-AK GIN-AK PNA-AK
# for r in "${redundancys[@]}"; do
#     python -m train.zinc --config train/configs/molhiv_sampling.yaml sampling.redundancy $r num_workers 14 device 5
# done

# later can add cifar 10

python -m train.cifar10 --config train/configs/cifar10_sampling.yaml num_workers 12  sampling.redundancy 4 sampling.random_rate 0.28  device 0 &
python -m train.cifar10 --config train/configs/cifar10_sampling.yaml num_workers 12  sampling.redundancy 5 sampling.random_rate 0.35  device 1 &

#   redundancy: 3
#   random_rate: 0.2 

