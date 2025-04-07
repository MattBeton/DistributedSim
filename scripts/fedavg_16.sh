#!/bin/bash

python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state
python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.75 --centered --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.5 --centered --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.25 --centered --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.0 --centered --sync_opt_state

#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.75
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.5
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.25
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.0

#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.75 --centered
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.5 --centered
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.25 --centered
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.0 --centered

#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.75 --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.5 --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.25 --sync_opt_state
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 0.0 --sync_opt_state
