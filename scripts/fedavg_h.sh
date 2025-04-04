#!/bin/bash
python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
