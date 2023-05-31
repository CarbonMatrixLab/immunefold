
local_world_size=1; python -m torch.distributed.launch\
    --nnodes 1 --node_rank 0 --nproc_per_node ${local_world_size} \
    --master_addr 127.0.0.1 --master_port 2222 \
    train.py  \
    --device gpu \
    --max_seq_len 256 \
    --batch_size 2 \
    --num_epoch 1024 \
    --warmup_steps 0 \
    --flat_steps 16384 \
    --learning_rate 0.0003 \
    --lr_decay poly \
    --prefix ./studies/15b_v1\
    --model_features ./config/config_data_pair.json \
    --model_config ./config/config_model_pair.json \
    --train_name_idx ../abdata_2023/sabdab/train_cluster.idx \
    --train_data ../abdata_2023/sabdab/npz \
    --general_data_gpu_ratio 0. \
    --train_general_name_idx ../ab_data/data/pdb50_v2/clean_bc40_cluster.idx \
    --train_general_data ../ab_data/data/pdb50_v2/data
