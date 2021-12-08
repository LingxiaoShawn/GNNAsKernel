
# PATTERN
python -m train.pattern  model.embs "(0, 1)" model.hops_dim 0  device 0 &

# MolHIV
python -m train.molhiv   model.embs "(0, 1)" model.hops_dim 0  device 1 &
python -m train.molhiv   model.embs "(0, )"  model.hops_dim 0  device 3 &
python -m train.molhiv   model.embs "(1, )"  model.hops_dim 0  device 4 &

# MolPCBA
python -m train.molpcba  model.embs "(0, 1)" model.hops_dim 0  device 2 &
python -m train.molpcba  model.embs "(1, )"  model.hops_dim 0  device 5 &

