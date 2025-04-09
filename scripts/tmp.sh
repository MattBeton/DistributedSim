#!/bin/bash
python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100 --damping 0.1 
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100 --damping 0.1 
python ../run/nanogpt_fedavg.py --num_nodes 4 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100 --damping 0.1 
python ../run/nanogpt_fedavg.py --num_nodes 2 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100 --damping 0.1 
python ../run/nanogpt_fedavg.py --num_nodes 1 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100 --damping 0.1 

python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 10.0 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 1.0 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 0.5 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 0.1 #--damping_meantrick

python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 10.0 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 1.0 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 0.5 #--damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 0.1 #--damping_meantrick


python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 10.0 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 1.0 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 0.5 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv1 --outer_interval 100 --damping 0.1 --damping_meantrick

python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 10.0 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 1.0 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 0.5 --damping_meantrick
python ../run/nanogpt_fedavg.py --num_nodes 8 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 100 --damping 0.1 --damping_meantrick



#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 50

#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method mean --outer_interval 100

#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 20
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 10
#python ../run/nanogpt_fedavg.py --num_nodes 16 --device_type cuda --devices 0 1 2 3 --dataset owt --model_size small --cosine_anneal --wandb_project tycho-test --batch_size 4 --outer_lr 1.0 --centered --sync_opt_state --merge_method curv0 --outer_interval 5

