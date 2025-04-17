#!/bin/bash

# no merge
#python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method nothing --outer_interval 2000 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001

# naive merge + sync
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --sync_opt_state --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --sync_opt_state --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 20 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --sync_opt_state --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 50 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --sync_opt_state --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 100 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --sync_opt_state --max_steps 2001

# naive merge
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 20 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 50 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 100 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001

# N
# python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method nothing --outer_interval 2001 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method nothing --outer_interval 2001 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method nothing --outer_interval 2001 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001

# H=1, N sweep
# python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 1 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001


# H=10, N sweep
# python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 6 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001
# python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 64 --local_minibatch_size 16 --outer_lr 1.0 --merge_method mean --outer_interval 10 --outer_warmup 1 --eval_interval 10 --warmup_steps 100 --max_steps 2001





