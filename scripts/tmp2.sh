#!/bin/bash

python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 8 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --outer_warmup 2000 --damping 0.01 --damping_meantrick --skip_embed_curv --forward_only

python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 8 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 500 --outer_warmup 2000


