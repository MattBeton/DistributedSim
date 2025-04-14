#!/bin/bash
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --merge_method mean --outer_interval 100 --outer_warmup 0 --damping 0.0001 --damping_meantrick --same_curv_data --lr 0.01 --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 0 --damping 0.0001 --damping_meantrick --same_curv_data --lr 0.01 --sync_opt_state

#python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --merge_method curv0 --outer_interval 100 --outer_warmup 0 --damping 0.0001 --damping_meantrick --same_curv_data --lr 0.01 --outer_warmup  1000




