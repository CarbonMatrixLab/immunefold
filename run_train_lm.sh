
local_world_size=1; python -m torch.distributed.launch\
    --nnodes 1 --node_rank 0 --nproc_per_node ${local_world_size} \
    --master_addr 127.0.0.1 --master_port 2222 \
    train_lm.py  \
    --device gpu \
    --max_seq_len 25 \
    --batch_size 2\
    --num_epoch 1024 \
    --warmup_steps 0 \
    --flat_steps 0 \
    --decay_steps 20000 \
    --learning_rate 0.0001 \
    --gradient_accumulation_it 4 \
    --prefix ./studies/lm_acc128_v1 \
    --restore_model_ckpt ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
    --model_features ./config/config_data_lm.json \
    --train_data ../oas_data/oas0.90/clu90_seq.fasta 
