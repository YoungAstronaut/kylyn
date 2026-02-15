#!/bin/bash
#SBATCH -J all_dft_decay_30_p_0.5_v_0.1
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100

echo "Input arguments"

set -ex

module load cuda/12.8

# 超算分配节点信息
echo "=== SLURM Job Info ==="
echo "Node List : ${SLURM_JOB_NODELIST:-NotAssigned}"
echo "Allocated CPU: $SLURM_CPUS_ON_NODE"
echo "Allocated GPU: ${SLURM_GPUS:-0}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# 获取实际运⾏节点（适⽤于已调度的作业）
if [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "Assigned Node: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
else
    echo "Assigned Node: $(hostname)"
fi
# 检查 GPU 设备
echo -e "\n=== GPU Status ==="
nvidia-smi -L || echo "GPU not detected"
echo "GPU Num Per Node $SLURM_GPUS_PER_NODE"
echo "Node Num $SLURM_JOB_NUM_NODES"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

unset ROCR_VISIBLE_DEVICES
export WANDB_MODE=offline
ulimit -u 65535

ray start --head --num-gpus=4

sleep 120

cd /home/jwangxgroup/yuhang/jyh/verl
bash recipe/mixed_train/run_mixed_train_7b_qwen2.5_base.sh \
  --devices 4