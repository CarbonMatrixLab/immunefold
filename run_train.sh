torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    ./train.py --config-name=train_carbonfold2
