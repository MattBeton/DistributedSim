#!/bin/bash
#python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0 --forward_only --skip_embed_curv --damping_meantrick
#python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0001 --forward_only --skip_embed_curv --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.0001 --forward_only --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.001 --forward_only --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.01 --forward_only --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 1.0 --forward_only --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 10.0 --forward_only --skip_embed_curv --damping_meantrick
pkill python

python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.01 --forward_only --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 10.0 --forward_only --skip_embed_curv --damping_meantrick
pkill python

pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0001 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.01 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 10.0 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.0001 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.01 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 1.0 --forward_only --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 10.0 --forward_only --skip_embed_curv 
pkill python

pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0001 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.01 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 10.0 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.0001 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.01 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 1.0 --skip_embed_curv --damping_meantrick
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 10.0 --skip_embed_curv --damping_meantrick
pkill python

pkill python

pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.0001 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 0.01 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 500 --damping 10.0 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.0001 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 0.01 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 1.0 --skip_embed_curv 
pkill python
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 500 --damping 10.0 --skip_embed_curv 
pkill python

pkill python
pkill python
pkill python
pkill python
