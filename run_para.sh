#!/bin/bash
#SBATCH -N 4 --gres=gpu:4 --qos=gpugpu -p vip_gpu_scx6023


module load anaconda/2021.11
module load compilers/cuda/11.6

conda init bash
source activate
conda deactivate
conda activate zhc

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO


### 获取每个节点的hostname
for i in `scontrol show hostnames`
do
	let k=k+1
	host[$k]=$i
	echo ${host[$k]}
done


local_world_size=4
nnodes=4

output_dir=./studies/esm_last8_v1
batch_size=2
general_data_gpu_ratio=0.0

cd /home/bingxing2/home/scx6023/zhang/AbFold2

python -m torch.distributed.launch\
    --nnodes=${nnodes} \
    --node_rank=0 \
    --nproc_per_node=${local_world_size} \
    --master_addr="${host[1]}" \
    --master_port="29501"  train.py  \
    --device gpu  \
    --max_seq_len 256 \
    --batch_size ${batch_size} \
    --num_epoch 1024 \
    --warmup_steps 1000 \
    --flat_steps 10000 \
    --learning_rate 0.0001 \
    --lr_decay poly \
    --checkpoint_it 10 \
    --prefix ${output_dir} \
    --restore_model_ckpt ../abdata_2023/esm2/abfold_from_esmfold.ckpt \
    --model_features ./config/config_data_pair.json \
    --model_config ./config/config_model_pair.json \
    --train_name_idx ../abdata_2023/sabdab/train_cluster.idx \
    --train_data ../abdata_2023/sabdab/npz \
    --general_data_gpu_ratio ${general_data_gpu_ratio} \
    --train_general_name_idx ../data_2023/pdb50_v2/clean_bc40_cluster.idx \
    --train_general_data ../data_2023/pdb50_v2/data   >>  train_rank0_${SLURM_JOB_ID}.log 2>&1 &a

### 使用srun 运行第二个节点
for r in `seq 2 ${nnodes}`
do	echo node ${r}	
	let rr=r-1
	srun -N 1 --gres=gpu:4 -w ${host[${r}]} \
		python -m torch.distributed.launch \
		--nnodes=${nnodes} \
		--node_rank=${rr} \
		--nproc_per_node=${local_world_size} \
		--master_addr="${host[1]}" \
		--master_port="29501" train.py  \
        --device gpu  \
        --max_seq_len 256 \
        --batch_size ${batch_size} \
        --num_epoch 1024 \
        --warmup_steps 1000 \
        --flat_steps 10000 \
        --learning_rate 0.0001 \
        --lr_decay poly \
        --checkpoint_it 10 \
        --prefix ${output_dir} \
        --restore_model_ckpt ../abdata_2023/esm2/abfold_from_esmfold.ckpt \
        --model_features ./config/config_data_pair.json \
        --model_config ./config/config_model_pair.json \
        --train_name_idx ../abdata_2023/sabdab/train_cluster.idx \
        --train_data ../abdata_2023/sabdab/npz \
        --general_data_gpu_ratio ${general_data_gpu_ratio} \
        --train_general_name_idx ../data_2023/pdb50_v2/clean_bc40_cluster.idx \
        --train_general_data ../data_2023/pdb50_v2/data  >> train_rank${rr}_${SLURM_JOB_ID}.log 2>&1 &
done

wait
