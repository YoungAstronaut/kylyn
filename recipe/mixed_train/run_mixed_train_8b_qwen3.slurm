#!/bin/bash
#SBATCH -J mixed_train_qwen2.5_sft0_rl1
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --calculate_rl)
            CALCULATE_RL="True"
            shift
            ;;
        --calculate_sft)
            CALCULATE_SFT="True"
            shift
            ;;
        --sft_coef)
            SFT_COEF="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

CALCULATE_RL=${CALCULATE_RL:-"False"}
CALCULATE_SFT=${CALCULATE_SFT:-"False"}
SFT_COEF="${SFT_COEF:-0.0}"
echo "Input arguments"
echo "CALCULATE_RL: $CALCULATE_RL"
echo "CALCULATE_SFT: $CALCULATE_SFT"
echo "SFT_COEF: $SFT_COEF"

set -ex

module load cuda/12.4

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

ray start --head

sleep 30

cd /home/jwangxgroup/yuhang/jyh/verl
bash recipe/mixed_train/run_mixed_train_8b_qwen3_rl_1_sft_0.sh \
  --devices 4 \
  --calculate_rl "$CALCULATE_RL" \
  --calculate_sft "$CALCULATE_SFT" \
  --sft_coef "$SFT_COEF"