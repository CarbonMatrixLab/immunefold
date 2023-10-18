#!/bin/bash
#SBATCH -N 4 --gres=gpu:4 --qos=gpugpu -p vip_gpu_scx6023


module load anaconda/2021.11 compilers/cuda/11.8
source activate py39_torch2.0
export PYTHONUNBUFFERED=1

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

output_dir=./studies/carbonfold_v1
batch_size=3

cd /home/bingxing2/home/scx6023/zhang/carbonmatrix

torchrun \
    --nnodes=${nnodes} \
    --nproc_per_node=${local_world_size} \
    --node_rank=0 \
    --master_addr="${host[1]}" \
    --master_port="29501" \
    ./train.py --config-name=train_carbonfold2  >>  train_rank0_${SLURM_JOB_ID}.log 2>&1 &

### 使用srun 运行第二个节点
for r in `seq 2 ${nnodes}`
do
    echo node ${r}
	let rr=r-1
	srun -N 1 --gres=gpu:4 -w ${host[${r}]} \
		torchrun \
		--nnodes=${nnodes} \
        --nproc_per_node=${local_world_size} \
		--node_rank=${rr} \
        --master_addr="${host[1]}" \
        --master_port="29501" \
        ./train.py --config-name=train_carbonfold2 >>  train_rank${rr}_${SLURM_JOB_ID}.log 2>&1 &
done

wait
