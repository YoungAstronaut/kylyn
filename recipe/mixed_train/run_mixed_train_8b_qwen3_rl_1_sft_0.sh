#!/usr/bin/env bash


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --devices)
            devices_num="$2"
            echo "设备数量: ${devices_num}"
            shift 2
            ;;
        --calculate_rl)
            calculate_rl_loss="$2"
            echo "计算 RL 损失: ${calculate_rl_loss}"
            shift 2
            ;;
        --calculate_sft)
            calculate_sft_loss="$2"
            echo "计算 SFT 损失: ${calculate_sft_loss}"
            shift 2
            ;;
        --sft_coef)
            sft_coef="$2"
            echo "SFT 损失系数: ${sft_coef}"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

calculate_rl_loss=${calculate_rl_loss:-"True"}
calculate_sft_loss=${calculate_sft_loss:-"False"}
sft_coef=${sft_coef:-"0.0"}

nnodes=1
n_gpus_per_node=${devices_num}
if [ "$devices_num" -lt 2 ]; then
    tensor_model_parallel_size=1
else
    tensor_model_parallel_size=2
fi
sp_size=1
echo "tensor_model_parallel_size: ${tensor_model_parallel_size}"
echo "sp_size: ${sp_size}"

#### 不改的目录
log_path=$HOME/jyh/verl/log
data_path=$HOME/jyh/verl/data
output_path=$HOME/jyh/verl/output


######
set -xeuo pipefail

project_name=luffy
experiment_name=mixed_sft_coef_0.051


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=8192
enable_overlong_buffer=True
overlong_buffer_len=8192
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
train_prompt_bsz=32  # train_batch_size
gen_prompt_bsz=8
n_resp_per_prompt=8
n_off_policy=0
ppo_mini_bsz=32

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Paths
model_path=$HOME/jyh/llm_models/Qwen/Qwen2.5-Math-7B-16k-think
embedding_model_path=$HOME/jyh/llm_models/Qwen/Qwen3-Embedding-8B
se_rollout_model_path=$HOME/jyh/llm_models/Qwen/Qwen2.5-7B-Instruct
default_local_dir=${output_path}/${project_name}/${experiment_name}
log_filename=${log_path}/${project_name}/${experiment_name}.log
mkdir -p ${log_path}/${project_name}
train_files=${data_path}/openr1.parquet
test_files=${data_path}/valid.parquet

# 算法
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance 相关超参
use_dynamic_bsz=True
infer_ppo_max_token_len=16384
offload=True

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
# PYTHONUNBUFFERED=1 \
python3 -m recipe.mixed_train.main_mixed_train \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_target_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.validation_shuffle=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.n_off_policy=${n_off_policy} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.calculate_sft_loss="${calculate_sft_loss}" \
    actor_rollout_ref.actor.sft_loss_coef="${sft_coef}" \
    actor_rollout_ref.actor.calculate_rl_loss="${calculate_rl_loss}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    embedding_worker.embedding_model.path="${embedding_model_path}" \
    se_rollout_worker.model.path="${se_rollout_model_path}" \
    se_rollout_worker.rollout.gpu_memory_utilization=0.80 \
    se_rollout_worker.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    se_rollout_worker.rollout.temperature=${temperature} \
    se_rollout_worker.rollout.top_p=${top_p} \
    se_rollout_worker.rollout.top_k="${top_k}" \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.nnodes="${nnodes}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.need_analyze_sft_grads=False \
    trainer.need_analyze_off_grads=False \
    trainer.analyze_gradients_freq=30 \
    trainer.llm_error_localization=False \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.resume_mode=auto 2>&1 | tee ${log_filename}