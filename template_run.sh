#!/bin/bash
ulimit -n 60000  # or a higher value if needed
log_id=$(date +"%Y%m%d_%H%M%S")
output_file="${log_id}.log"

export downstream_root='./datasets'
export task_root='./tasks'
export CUDA_VISIBLE_DEVICES=2

python train_ssl.py --pretrain_data 'cora' \
 --pretrain_data_ids -1 --pretrain_data_size -1 \
 --opt 'adam' --weight_decay 1e-5 --alpha 3.0 \
 --linear_lr 0.01 --linear_l2 1e-4 \
 --task 'vgae' --n_shots 5 --n_tasks 5 --eval_data_seed 0 \
 --scheduler 'constant' \
 --encoder_name 'GCN' --n_layers 2 --hidden_dim 384 \
 --n_head 4 --attn_drop 0.1 --norm 'none' --activation 'relu' \
 --num_workers 0 \
 --peak_lr 0.001 --warmup_steps -1 --epochs 100 --dropout 0.2 \
 --eval_data_names 'cora' \
 --eval_step 10 --save_step 10 \
 --log_id "$log_id" \
