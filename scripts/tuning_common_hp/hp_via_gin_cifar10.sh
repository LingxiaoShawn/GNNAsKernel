gpu=3
hiddens=(64 128 256)
layers=(2 4 5 6)
drops=(0.0 0.5)
pools=(add mean)


for drop in "${drops[@]}"; do
for hidden in "${hiddens[@]}"; do
for layer in "${layers[@]}"; do
    python -m train.cifar10 model.hidden_size $hidden model.num_layers $layer model.mini_layers 0 train.dropout $drop device $gpu subgraph.online False
done
done
done