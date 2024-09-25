source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate esmfold
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port 7077 \
    ./train.py --config-name=train_abfold
