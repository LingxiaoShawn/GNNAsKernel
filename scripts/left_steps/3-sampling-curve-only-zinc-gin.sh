redundancys=(1 2 3 4 5)
# 1. zinc, GCN-AK GIN-AK PNA-AK
for r in "${redundancys[@]}"; do
    python -m train.zinc --config train/configs/molhiv_sampling.yaml sampling.redundancy $r
done

# later can add cifar 10

