
local_world_size=1; python -m torch.distributed.launch\
    --nnodes 1 --node_rank 0 --nproc_per_node ${local_world_size} \
    --master_addr 127.0.0.1 --master_port 2223 \
    train_stage1.py  \
    --device gpu \
    --gpu_list 0 \
    --max_seq_len 256 \
    --batch_size 1 \
    --num_epoch 1024 \
    --warmup_steps 0 \
    --flat_steps 16384 \
    --decay_steps 16384 \
    --learning_rate 0.0001 \
    --gradient_accumulation_it 2\
    --prefix ./studies/v1\
    --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm2.ckpt \
    --restore_esm2_model ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
    --model_features ./config/config_data_stage1.json \
    --model_config ./config/config_model_stage1.json \
    --train_name_idx ../oas_data/stage1_pdb.list \
    --train_data ../oas_data/
