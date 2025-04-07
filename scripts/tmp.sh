#!/bin/bash

python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100
