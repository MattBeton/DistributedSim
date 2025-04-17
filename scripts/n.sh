#!/bin/bash


# H=1, N sweep
python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state


# H=5, N sweep
python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 5 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 5 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 5 --warmup_steps 100 --max_steps 2001 --sync_opt_state

# H=10, N sweep
python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001 --sync_opt_state





