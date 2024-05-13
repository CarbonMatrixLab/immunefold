source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate esm_flash_attn
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port 7070 \
    ./train.py --config-name=train_tcrfold
