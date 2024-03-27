# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p -3.0 model.gnn_type PPGN device 0 &
# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p -1.0 model.gnn_type PPGN device 1 &
# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p -0.3 model.gnn_type PPGN device 2 &
# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p -0.1 model.gnn_type PPGN device 3 &
# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p 0.1  model.gnn_type PPGN device 4 &
# python -m train.zinc num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p 0.3  model.gnn_type PPGN device 5 &


# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 8 subgraph.FGSD_p -3.0 model.gnn_type PPGN device 0 &
# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 16 subgraph.FGSD_p -1.0  device 1 train.epochs 40&
# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 16 subgraph.FGSD_p -0.3  device 2 train.epochs 40&
# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 16 subgraph.FGSD_p -0.1  device 3 train.epochs 40&
# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 16 subgraph.FGSD_p 0.1   device 4 train.epochs 40&
# python -m train.sr25 num_workers 16 subgraph.FGSD True subgraph.FGSD_k 16 subgraph.FGSD_p 0.3   device 5 train.epochs 40&




python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 8 subgraph.FGSD_p -1.0  device 1 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 8 subgraph.FGSD_p -0.3  device 2 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 8 subgraph.FGSD_p -0.1  device 3 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 8 subgraph.FGSD_p 0.1   device 4 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 8 subgraph.FGSD_p 0.3   device 5 &

wait 

python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.15 subgraph.FGSD_p -1.0  device 1 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.15 subgraph.FGSD_p -0.3  device 2 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.15 subgraph.FGSD_p -0.1  device 3 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.15 subgraph.FGSD_p 0.1   device 4 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.15 subgraph.FGSD_p 0.3   device 5 &

wait

python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.3 subgraph.FGSD_p -1.0  device 1 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.3 subgraph.FGSD_p -0.3  device 2 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.3 subgraph.FGSD_p -0.1  device 3 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.3 subgraph.FGSD_p 0.1   device 4 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_q 0.3 subgraph.FGSD_p 0.3   device 5 &

wait 
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 13 subgraph.FGSD_p -1.0  device 1 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 13 subgraph.FGSD_p -0.3  device 2 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 13 subgraph.FGSD_p -0.1  device 3 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 13 subgraph.FGSD_p 0.1   device 4 &
python -m train.zinc_2 train.runs 2 num_workers 16  subgraph.FGSD_k 13 subgraph.FGSD_p 0.3   device 5 &