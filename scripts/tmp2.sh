#!/bin/bash
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method soap --outer_interval 10 --outer_warmup 0 --eval_interval 10 --warmup_steps 100 --damping 0.1 #--sync_opt_state #--sync_opt_state # --skip_embed_curv #--forward_only



