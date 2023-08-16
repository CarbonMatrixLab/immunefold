#!/bin/bash
#SBATCH -N 4 --gres=gpu:4 --qos=gpugpu -p vip_gpu_scx6023

module load anaconda/2021.11
module load compilers/cuda/11.6

conda init bash
source activate
conda deactivate
conda activate esmfold2

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

gradient_accumulation_it=8
decay_steps=40000

output_dir=./studies/stage1_all_v1
batch_size=1

cd /home/bingxing2/home/scx6023/zhang/carbonmatrix

python -m torch.distributed.launch\
    --nnodes=${nnodes} \
    --node_rank=0 \
    --nproc_per_node=${local_world_size} \
    --master_addr="${host[1]}" \
    --master_port="29501"  train_stage1.py  \
    --device gpu \
    --max_seq_len 256 \
    --batch_size ${batch_size} \
    --num_epoch 1024 \
    --warmup_steps 0 \
    --flat_steps 0 \
    --decay_steps ${decay_steps} \
    --learning_rate 0.0001 \
    --gradient_accumulation_it ${gradient_accumulation_it} \
    --prefix ${output_dir} \
    --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm2.ckpt \
    --restore_esm2_model ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
    --model_features ./config/config_data_stage1.json \
    --model_config ./config/config_model_stage1.json \
    --train_name_idx ../oas_data/shuffled_train_pdb.list \
    --train_data ../oas_data/ >>  train_rank0_${SLURM_JOB_ID}.log 2>&1 &

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
		--master_port="29501" train_stage1.py  \
        --device gpu \
        --max_seq_len 256 \
        --batch_size ${batch_size} \
        --num_epoch 1024 \
        --warmup_steps 0 \
        --flat_steps 0 \
        --decay_steps ${decay_steps} \
        --learning_rate 0.0001 \
        --gradient_accumulation_it ${gradient_accumulation_it} \
        --prefix ${output_dir} \
        --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm2.ckpt \
        --restore_esm2_model ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
        --model_features ./config/config_data_stage1.json \
        --model_config ./config/config_model_stage1.json \
        --train_name_idx ../oas_data/shuffled_train_pdb.list \
        --train_data ../oas_data/ >> train_rank${rr}_${SLURM_JOB_ID}.log 2>&1 &
done

wait
