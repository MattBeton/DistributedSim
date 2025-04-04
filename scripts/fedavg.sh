#!/bin/bash
python ../run/nanogpt_diloco.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4
