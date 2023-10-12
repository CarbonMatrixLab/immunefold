torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./train.py --config-name=train_diffuser
