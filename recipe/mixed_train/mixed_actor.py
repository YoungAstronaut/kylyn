# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""
import hashlib
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import distributed as dist, Tensor
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import ceildiv, get_seqlen_balanced_partitions, roundup_divisible, \
    rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

# if is_cuda_available:
#     from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
# elif is_npu_available:
#     from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EPS = 1e-12

def _to_f32(x):
    return x.detach().to(torch.float32)

def gather_sft_tensor(data, prompt_length, tgt_inputs_length, pad_token_id):
    sft_responses = data['tgt_input_ids']
    sft_input_ids = torch.cat([data["input_ids"][:, :prompt_length], sft_responses], dim=-1)
    sft_attention_mask = (sft_input_ids != pad_token_id).int()
    sft_position_ids = data["position_ids"]
    print('sft input ids: ', sft_input_ids.shape)
    print('sft attention mask: ', sft_attention_mask.shape)
    print('sft position ids: ', sft_position_ids.shape)
    print('sft responses: ', sft_responses.shape)

    return sft_input_ids, sft_attention_mask, sft_position_ids, sft_responses

def calculate_cos_and_angle(a, b):
    n1 = a.norm().clamp_min(EPS)
    n2 = b.norm().clamp_min(EPS)
    cos = ((a * b).sum() / (n1 * n2)).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.rad2deg(torch.acos(cos))
    return cos, angle

def _safe_angle_from_cos(cos_val: float):
    if cos_val is None:
        return None
    t = torch.tensor(cos_val, dtype=torch.float32, device=get_device_id())
    t = t.clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(t)).item()

def split_on_policy_batch(
        batch: TensorDict
) -> (TensorDict, TensorDict):
    batch_size = batch.batch_size[0]
    on_policy_partitions = []
    off_policy_partitions = []
    for i in range(batch_size):
        valid_mask = batch['on_policy_mask'][i:i + 1].sum()
        assert valid_mask == 0 or valid_mask == batch[i:i+1]['on_policy_mask'].shape[-1], \
            f'the on policy mask is not all zeros neither all ones. valid mask: {valid_mask}'
        if valid_mask > 0:
            on_policy_partitions.append(batch[i:i + 1])
        else:
            off_policy_partitions.append(batch[i:i + 1])
    return torch.cat(on_policy_partitions) if len(on_policy_partitions) > 0 else None, \
        torch.cat(off_policy_partitions) if len(off_policy_partitions) > 0 else None

def analyze_gradients(sft_grads, rl_grads):
    results = {}
    for name in sft_grads.keys():
        sft_grad = sft_grads[name]
        rl_grad = rl_grads[name]
        
        # 1. 计算L2范数
        sft_norm = torch.norm(sft_grad, p=2).item()
        rl_norm = torch.norm(rl_grad, p=2).item()
        norm_ratio = sft_norm / rl_norm
        
        # 2. 计算余弦相似度
        cosine_sim = torch.cosine_similarity(
            sft_grad.flatten(), 
            rl_grad.flatten(), 
            dim=0
        ).item()
        
        # 3. 记录结果
        results[name] = {
            "sft_norm": sft_norm,
            "rl_norm": rl_norm,
            "norm_ratio": norm_ratio,
            "cosine_sim": cosine_sim,
            "conflict": "严重" if cosine_sim < -0.7 else 
                       "高" if cosine_sim < 0.3 else 
                       "中" if cosine_sim < 0.7 else "低"
        }
    
    # 打印层级别分析报告
    print(f"{'Layer':<20} {'SFT Norm':<10} {'RL Norm':<10} {'Ratio':<8} {'CosSim':<8} {'Conflict':<6}")
    for name, data in results.items():
        print(f"{name:<20} {data['sft_norm']:<10.4f} {data['rl_norm']:<10.4f} {data['norm_ratio']:<8.2f} {data['cosine_sim']:<8.2f} {data['conflict']:<6}")
    
    return results

def compute_sft_loss(log_prob, mask):
    assert log_prob.shape == mask.shape, f'log_prob shape {log_prob.shape} does not match mask shape {mask.shape}'
    # valid_tokens = torch.sum(mask).item()
    # print(f"SFT有效token数量: {valid_tokens}")
    return -torch.sum(log_prob * mask) / torch.sum(mask)

def shape_on_policy(shape_strategy: str,
                    old_log_prob: Tensor,
                    log_prob: Tensor,
                    policy_reshape_pow_exp: float = 0.5,
                    policy_reshape_weight: float = 1.0,
                    eps: float = 1e-8):
    # 基础量：log 比率
    log_ratio = log_prob - old_log_prob
    # 防溢出/下溢
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    # 返回 (ratio_like, is_multiplicative_ratio)
    if shape_strategy == "no_reshape":
        ratio = torch.exp(log_ratio)
        return ratio, True
    elif shape_strategy == "logp":
        # 加性整形（不是乘性 ratio）
        ratio = policy_reshape_weight * log_ratio
        return ratio, False
    elif shape_strategy == "p_logp":
        r = torch.exp(log_ratio)
        ratio = r + policy_reshape_weight * log_ratio
        return ratio, True  # 仍然有 ratio 主体
    elif shape_strategy == "square_root":
        ratio = torch.sqrt(torch.exp(log_ratio))
        return ratio, True
    elif shape_strategy == "pow":
        ratio = torch.pow(torch.exp(log_ratio), policy_reshape_pow_exp)
        return ratio, True
    elif shape_strategy in {"p_div_p_0.1", "p_div_p_0.5", "p_div_p_0.3"}:
        c = float(shape_strategy.split("_")[-1])
        p = torch.exp(log_prob)            # (0, 1]
        p_old = torch.exp(old_log_prob)    # (0, 1]
        f = p / (p + c)
        f_old = p_old / (p_old + c)
        ratio = (f + eps) / (f_old + eps)  # 防 0/0
        return ratio, True
    else:
        raise ValueError(f"Invalid on_policy_reshape: {shape_strategy}")

def shape_off_policy(shape_strategy: str,
                     old_log_prob: Tensor,
                     log_prob: Tensor,
                     policy_reshape_weight: float = 1.0,
                     policy_reshape_pow_exp: float = 0.5,
                     eps: float = 1e-8):
    # 这里不是严格 ratio，更多是“提升该 token 概率”的系数
    off_ratio = torch.exp(log_prob)  # (0, 1]
    if shape_strategy == "no_reshape":
        pass
    elif shape_strategy == "logp":
        off_ratio = policy_reshape_weight * log_prob
    elif shape_strategy == "p_logp":
        off_ratio = policy_reshape_weight * log_prob + off_ratio
    elif shape_strategy == "square_root":
        off_ratio = torch.sqrt(off_ratio)
    elif shape_strategy == "p_div_p_0.1":
        off_ratio = off_ratio / (off_ratio + 0.1)
    elif shape_strategy == "p_div_p_0.5":
        off_ratio = off_ratio / (off_ratio + 0.5)
    elif shape_strategy == "p_div_p_0.3":
        off_ratio = off_ratio / (off_ratio + 0.3)
    elif shape_strategy == "pow":
        off_ratio = torch.pow(off_ratio, policy_reshape_pow_exp)
    else:
        raise ValueError(f"Invalid off_policy_reshape: {shape_strategy}")
    return off_ratio  # ← 必须返回

def compute_token_mixed_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange_low,
    cliprange_high,
    on_policy_reshape="no_reshape",
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    on_policy_mask=None,
    off_policy_reshape="no_reshape",
    off_policy_reshape_weight=1.0,
    off_policy_reshape_pow_exp=0.5,
    off_max_clip=None,
    off_min_clip=None,
    all_max_clip=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    # 1) 形状与基本检查
    assert log_prob.shape == old_log_prob.shape, f'log_prob shape {log_prob.shape} != old_log_prob shape {old_log_prob.shape}'
    assert log_prob.shape == advantages.shape,   f'log_prob shape {log_prob.shape} != advantages shape {advantages.shape}'
    assert log_prob.shape == eos_mask.shape,     f'log_prob shape {log_prob.shape} != eos_mask shape {eos_mask.shape}'

    # 统一设备/类型的零
    def _zero_like_scalar(x: Tensor):
        return x.new_zeros(())

    # 把 on_policy_mask 处理成 bool mask
    has_on_mask = on_policy_mask is not None
    if has_on_mask:
        assert on_policy_mask.shape == eos_mask.shape, "on_policy_mask shape mismatch"
        if on_policy_mask.dtype != torch.bool:
            on_policy_mask = on_policy_mask.bool()
        on_mask = on_policy_mask & eos_mask.bool()
        off_mask = (~on_policy_mask) & eos_mask.bool()
    else:
        on_mask = eos_mask.bool()
        off_mask = None  # 没 off-policy

    # 2) 近似 KL（日志用途）：old - new
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # 3) on-policy ratio 与（可选）裁剪
    ratio, is_multiplicative = shape_on_policy(
        on_policy_reshape, old_log_prob, log_prob,
        policy_reshape_pow_exp=on_policy_reshape_pow_exp,
        policy_reshape_weight=on_policy_reshape_weight
    )

    if (all_max_clip is not None) and is_multiplicative:
        ratio = torch.clamp(ratio, max=all_max_clip)

    on_pg_losses = -advantages * ratio

    if (not loss_remove_clip) and is_multiplicative:
        ratio_clamped = torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
        on_pg_losses2 = -advantages * ratio_clamped
        # 注意：clipfrac 统计仅对 on-policy token
        on_pg_clipfrac = verl_F.masked_mean((on_pg_losses2 > on_pg_losses).float(), on_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
    else:
        on_pg_clipfrac = _zero_like_scalar(log_prob)

    # 仅对 on-policy 区域统计 on_pg_loss
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_mask)

    # 4) off-policy（如果有）
    if has_on_mask:
        off_ratio = shape_off_policy(
            off_policy_reshape, old_log_prob, log_prob,
            policy_reshape_weight=off_policy_reshape_weight,
            policy_reshape_pow_exp=off_policy_reshape_pow_exp
        )
        if all_max_clip is not None:
            off_ratio = torch.clamp(off_ratio, max=all_max_clip)
        if off_max_clip is not None:
            off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        if off_min_clip is not None:
            off_ratio = torch.clamp(off_ratio, min=off_min_clip)

        off_pg_losses = -advantages * off_ratio
        off_pg_loss = verl_F.masked_mean(off_pg_losses, off_mask)
        # 混合
        pg_losses = on_pg_losses * on_mask.float() + off_pg_losses * off_mask.float()
    else:
        off_pg_loss = _zero_like_scalar(log_prob)
        pg_losses = on_pg_losses

    # 5) 总 loss 的归一化
    if loss_remove_token_mean:
        # 只按 batch 平均（去掉 token 级平均）
        batch_size = log_prob.size(0)
        pg_loss = (pg_losses * eos_mask).sum() / max(batch_size, 1)
    else:
        # 标准 masked mean
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return pg_loss, on_pg_loss, on_pg_clipfrac, off_pg_loss, ppo_kl

    # return {
    #     "pg_loss": pg_loss,
    #     "off_pg_loss": off_pg_loss,
    #     "on_pg_loss": on_pg_loss,
    #     "off_pg_clipfrac": off_pg_clipfrac,
    #     "on_pg_clipfrac": on_pg_clipfrac,
    #     "ppo_kl": ppo_kl,
    #     "off_policy_prob": off_policy_prob,
    #     "on_policy_prob": on_policy_prob,
    #     "off_ratio_mean": off_ratio_mean,
    #     "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
    #     "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    # }

def rearrange_micro_batches_with_targets(
    batch,
    max_token_len,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    with_sft=True,
    with_rl=True
):
    """
    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        with_sft:
        with_rl:

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    # this is per local micro_bsz
    max_seq_len = batch["attention_mask"].shape[-1]
    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )
    if not with_sft and with_rl:
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    elif with_rl and with_sft:
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)+batch["tgt_attention_mask"].sum(dim=1)
    elif with_sft and not with_rl:
        seq_len_effective: torch.Tensor = batch["tgt_attention_mask"].sum(dim=1)
    else:
        raise ValueError("with_sft and with_rl cannot be both False")
    total_seqlen = seq_len_effective.sum().item()
    print(f'total seq len {total_seqlen}')
    # print(f'seq_len_effective: {seq_len_effective}')
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    seq_len_effective = seq_len_effective.tolist()
    assert num_micro_batches <= len(seq_len_effective)

    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = []
        for idx in partition:
            curr_micro_batch.append(batch[idx : idx + 1])
        curr_micro_batch = torch.cat(curr_micro_batch)

        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx

def convert_data(data):
    if isinstance(data, DataProto):
        result = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
    elif isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(get_device_id())
            elif k == "multi_modal_inputs" and v is not None:
                result[k] = [
                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                ]
            else:
                result[k] = v
    else:
        result = data.to(get_device_id())  # actor device is cpu when using offload
    return result


def _sanitize_name(name: str) -> str:
    # 便于作为文件名（尽量短）
    base = name.replace("/", "_").replace(".", "_").replace(" ", "_")
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}"


class MixedTrainParallelPPOActor(DataParallelPPOActor):

    def _precondition_grad(self, g, p, param_eps):
        """
        用 Adam/AdamW 的二阶动量对梯度做预条件： g / (sqrt(v) + eps)
        - v 优先取 amsgrad 的 max_exp_avg_sq，否则取 exp_avg_sq
        - 若拿不到 state（比如第一步），返回 None 表示无法预条件
        """
        if g is None:
            return None
        state = self.actor_optimizer.state.get(p, None)
        if not state:
            return None
        v = state.get("max_exp_avg_sq", None)
        if v is None:
            v = state.get("exp_avg_sq", None)
        if v is None:
            return None
        eps = param_eps.get(p, 1e-8)
        g32 = _to_f32(g)
        v32 = _to_f32(v)
        # 设备对齐
        if v32.device != g32.device:
            v32 = v32.to(g32.device)
        denom = torch.sqrt(v32).add(eps)
        # 避免不必要的显存放大
        return g32 / denom
    
    def _global_cos_and_angle(self, vec_a: torch.Tensor, vec_b: torch.Tensor, EPS: float = 1e-12, group=None):
        # vec_* 一维或可展平向量（已在同一 device）
        a = vec_a.reshape(-1).float()
        b = vec_b.reshape(-1).float()
        # 局部部分和
        local_dot   = (a * b).sum()
        local_n1_sq = (a * a).sum()
        local_n2_sq = (b * b).sum()
        # 分布式聚合：把所有 rank 的部分和相加得到全局和
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_dot,   op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(local_n1_sq, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(local_n2_sq, op=dist.ReduceOp.SUM, group=group)
        # 用全局的和计算余弦与角度/范数
        n1 = local_n1_sq.sqrt().clamp_min(EPS)
        n2 = local_n2_sq.sqrt().clamp_min(EPS)
        cos = (local_dot / (n1 * n2 + EPS)).clamp(-1 + 1e-6, 1 - 1e-6)
        angle = torch.rad2deg(torch.acos(cos))
        return cos, angle, n1, n2

    def _grab_grads_recompute(
            self,
            which: str,  # "sft" / "pg"
            batch_data: TensorDict,
            response: torch.Tensor,
            temperature: float,
            clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
            *,
            dp_no_sync: bool = False,  # <== 关键：False 表示做 reduce-scatter，得到“分片梯度”
            return_grads: bool = False,  # 多数情况下我们直接落盘，不返回大列表
            save_dir: str | None = None,  # 不为 None 时落盘
            save_cpu_dtype: torch.dtype = torch.float16,
            per_param_files: bool = True,
            save_optimizer_state: bool = False,  # 如需离线做预条件余弦，建议 True
            step: int = 0,
            mini_batch_idx: int = 0,
            on_policy_mask: torch.Tensor | None = None,
    ):
        """
        计算一次 SFT/PG 的纯净梯度（与优化器状态一致的“分片梯度”），并可选直接落盘。
        - 当 save_dir 不为 None 时：流式拷贝到 CPU 并保存文件；返回 manifest（小字典）
        - 当 return_grads 为 True 时：返回 grads(list[Tensor|None])（不建议与 save_dir 同时用）
        """
        assert which in ("sft", "pg", "on_pg", "off_pg")
        assert not (save_dir and return_grads), "建议二选一：要么落盘，要么返回列表"

        # FSDP/DDP 同步控制（False -> 同步 -> shard grads）
        no_sync_ctx = getattr(self.actor_module, "no_sync", None)
        sync_ctx = no_sync_ctx() if (dp_no_sync and no_sync_ctx is not None) else nullcontext()

        self.actor_optimizer.zero_grad(set_to_none=True)

        with sync_ctx:
            _, all_log_prob_tmp = self._forward_micro_batch(
                micro_batch=batch_data, temperature=temperature
            )
            if which == "sft":
                loss_scalar = compute_sft_loss(all_log_prob_tmp, response)
            else:
                loss_scalar, _, _, _, _ = compute_token_mixed_policy_loss(
                    old_log_prob=batch_data["old_log_probs"],
                    log_prob=all_log_prob_tmp,
                    advantages=batch_data["advantages"],
                    eos_mask=response,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                    off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                    all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                    on_policy_reshape=self.config.on_policy_reshape,
                    on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                    on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                    off_policy_reshape=self.config.off_policy_reshape,
                    off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                    off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                    loss_remove_token_mean=self.config.loss_remove_token_mean,
                    loss_remove_clip=self.config.loss_remove_clip,
                    on_policy_mask=on_policy_mask,
                )
            loss_scalar.backward()  # 在 dp_no_sync=False 下，这里会做 reduce-scatter，得到“分片梯度”

        params_named = [(n, p) for n, p in self.actor_module.named_parameters() if p.requires_grad]
        names, params = zip(*params_named) if params_named else ([], [])

        # 直接返回 grads（不落盘）
        if return_grads and not save_dir:
            grads = []
            for p in params:
                g = p.grad
                if g is None:
                    grads.append(None)
                else:
                    # 分片梯度体量较小，可选：直接搬到 CPU 再返回
                    grads.append(g.detach().to("cpu", dtype=save_cpu_dtype))
            self.actor_optimizer.zero_grad(set_to_none=True)
            return grads

        # ====== 落盘模式 ======
        if save_dir:
            # 组织目录：{save_dir}/step_{global_step}/{which}/rank_{rank}
            rank = torch.distributed.get_rank() if (
                        torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
            world_size = torch.distributed.get_world_size() if (
                        torch.distributed.is_available() and torch.distributed.is_initialized()) else 1
            root = Path(save_dir) / f"step_{step:03d}" / f"batch_idx_{mini_batch_idx:04d}" / which / f"rank_{rank:03d}"
            print(f"[MixedActor] 落盘目录：{root}")
            root.mkdir(parents=True, exist_ok=True)

            manifest = {
                "which": which,
                "step": step,
                "rank": rank,
                "world_size": world_size,
                "dtype": str(save_cpu_dtype).replace("torch.", ""),
                "dp_no_sync": bool(dp_no_sync),
                "timestamp": int(time.time()),
                "params": [],  # 每个条目包含 name / file / shape / numel
            }

            # 可选：把优化器的 v（exp_avg_sq / max_exp_avg_sq）也一起保存（用于离线预条件余弦）
            def _get_v_state(p):
                if not save_optimizer_state:
                    return None
                st = self.actor_optimizer.state.get(p, None)
                if not st:
                    return None
                return st.get("max_exp_avg_sq", None) or st.get("exp_avg_sq", None)

            # 流式拷贝并保存
            for idx, (name, p) in enumerate(params_named):
                g = p.grad
                if g is None:
                    continue
                # 搬到 CPU（分片梯度 -> 体量较小）
                g_cpu = g.detach().to("cpu", dtype=save_cpu_dtype, non_blocking=False).contiguous()

                safe = _sanitize_name(name)
                if per_param_files:
                    # 每参数单独一个 .pt
                    grad_path = root / f"param_{idx:05d}__{safe}.pt"
                    torch.save(g_cpu, grad_path)
                    v_path = None
                    if save_optimizer_state:
                        v = _get_v_state(p)
                        if v is not None:
                            v_cpu = v.detach().to("cpu", dtype=torch.float32, non_blocking=False).contiguous()
                            v_path = root / f"param_{idx:05d}__{safe}__v.pt"
                            torch.save(v_cpu, v_path)
                    manifest["params"].append({
                        "idx": idx,
                        "name": name,
                        "file": grad_path.name,
                        "v_file": (v_path.name if v_path else None),
                        "shape": list(g_cpu.shape),
                        "numel": int(g_cpu.numel()),
                    })
                else:
                    # 一个 rank 一个大文件（不推荐，难以部分恢复）
                    # 这里就跳过；你也可以拼到一个 dict 里再 save
                    pass

                # 立刻释放 GPU 端 grad（以免后续显存峰值）
                p.grad = None

            # 写 manifest
            (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

            # 清理
            self.actor_optimizer.zero_grad(set_to_none=True)
            return manifest

        # 兜底：既不返回也不保存
        self.actor_optimizer.zero_grad(set_to_none=True)
        return None

    def analyze_grads(self, grads_sft, grads_pg, dp_group=None):
        names = [n for n, p in self.actor_module.named_parameters() if p.requires_grad]
        params = [p for _, p in self.actor_module.named_parameters() if p.requires_grad]

        # 选择一个 GPU 设备用于 all_reduce（小向量）
        device0 = next(self.actor_module.parameters()).device

        # eps map
        param_eps = {}
        for g in self.actor_optimizer.param_groups:
            e = g.get("eps", 1e-8)
            for p in g["params"]:
                param_eps[p] = e

        per_param_rows = []
        idx_valid = []
        idx_valid_pre = []

        # 这些列表放 **Python float** 或 CPU float
        local_dot, local_n1sq, local_n2sq = [], [], []
        local_dot_pre, local_n1sq_pre, local_n2sq_pre = [], [], []

        with torch.no_grad():
            for i, (name, p, g1, g2) in enumerate(zip(names, params, grads_sft, grads_pg)):
                if g1 is None or g2 is None:
                    per_param_rows.append({
                        "name": self.pretty_fsdp_flat_name(name),
                        "cos": None, "angle_deg": None,
                        "norm_sft": 0.0, "norm_pg": 0.0,
                        "cos_pre": None, "angle_pre_deg": None,
                    })
                    continue

                # g1/g2 都在 CPU（上一个函数就这么返回的）
                a = g1.view(-1).to(torch.float32)
                b = g2.view(-1).to(torch.float32)
                # 本地标量（Python float）
                local_dot.append(float((a * b).sum().item()))
                local_n1sq.append(float((a * a).sum().item()))
                local_n2sq.append(float((b * b).sum().item()))
                idx_valid.append(i)

                # 预条件（把 v 拉到 CPU，按需逐层处理，内存峰值很低）
                state = self.actor_optimizer.state.get(p, None)
                v = None
                if state:
                    v = state.get("max_exp_avg_sq", None) or state.get("exp_avg_sq", None)
                if v is not None:
                    v_cpu = v.detach().to(device="cpu", dtype=torch.float32)
                    denom = v_cpu.sqrt().add(param_eps.get(p, 1e-8))
                    pa = a / denom
                    pb = b / denom
                    local_dot_pre.append(float((pa * pb).sum().item()))
                    local_n1sq_pre.append(float((pa * pa).sum().item()))
                    local_n2sq_pre.append(float((pb * pb).sum().item()))
                    idx_valid_pre.append(i)
                else:
                    idx_valid_pre.append(None)

            # ===== 把 CPU 的标量列表打包成 **很小的 CUDA 向量** 再做 all_reduce =====
            def _reduce_float_list(xs):
                if len(xs) == 0:
                    return None
                t = torch.tensor(xs, device=device0, dtype=torch.float32)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=dp_group)
                return t

            dot_g = _reduce_float_list(local_dot)
            n1sq_g = _reduce_float_list(local_n1sq)
            n2sq_g = _reduce_float_list(local_n2sq)

            dot_pre_g = _reduce_float_list(local_dot_pre)
            n1sq_pre_g = _reduce_float_list(local_n1sq_pre)
            n2sq_pre_g = _reduce_float_list(local_n2sq_pre)

            block_cos_list, precond_block_cos_list = [], []
            EPS_ = 1e-12

            # 普通版
            k = 0
            for row in per_param_rows:
                if row["cos"] is not None:
                    n1 = n1sq_g[k].sqrt().clamp_min(EPS_)
                    n2 = n2sq_g[k].sqrt().clamp_min(EPS_)
                    cos = (dot_g[k] / (n1 * n2 + EPS_)).clamp(-1 + 1e-6, 1 - 1e-6)
                    ang = torch.rad2deg(torch.acos(cos))
                    row["cos"] = float(cos.item())
                    row["angle_deg"] = float(ang.item())
                    row["norm_sft"] = float(n1.item())
                    row["norm_pg"] = float(n2.item())
                    block_cos_list.append(row["cos"])
                    k += 1

            # 预条件版
            k = 0
            for i, row in enumerate(per_param_rows):
                if idx_valid_pre[i] is not None and dot_pre_g is not None:
                    pn1 = n1sq_pre_g[k].sqrt().clamp_min(EPS_)
                    pn2 = n2sq_pre_g[k].sqrt().clamp_min(EPS_)
                    cos_p = (dot_pre_g[k] / (pn1 * pn2 + EPS_)).clamp(-1 + 1e-6, 1 - 1e-6)
                    ang_p = torch.rad2deg(torch.acos(cos_p))
                    row["cos_pre"] = float(cos_p.item())
                    row["angle_pre_deg"] = float(ang_p.item())
                    precond_block_cos_list.append(row["cos_pre"])
                    k += 1

            # 汇总指标（同你现有逻辑）
            mini_batch_metrics = {}
            for r in per_param_rows:
                layer = r["name"]
                mini_batch_metrics[f"actor/pp_cos_{layer}"] = r["cos"]
                mini_batch_metrics[f"actor/pp_ang_{layer}"] = r["angle_deg"]
                mini_batch_metrics[f"actor/pp_n_sft_{layer}"] = r["norm_sft"]
                mini_batch_metrics[f"actor/pp_n_pg_{layer}"] = r["norm_pg"]
                if r["cos_pre"] is not None:
                    mini_batch_metrics[f"actor/pp_cos_pre_{layer}"] = r["cos_pre"]
                    mini_batch_metrics[f"actor/pp_ang_pre_{layer}"] = r["angle_pre_deg"]

            def _ang_from_cos(c):
                if c is None: return None
                t = torch.tensor(c).clamp(-1 + 1e-6, 1 - 1e-6)
                return float(torch.rad2deg(torch.acos(t)).item())

            if len(block_cos_list) > 0:
                gb = float(sum(block_cos_list) / len(block_cos_list))
                mini_batch_metrics["actor/global_block_cos"] = gb
                mini_batch_metrics["actor/global_block_ang_deg"] = _ang_from_cos(gb)
            else:
                mini_batch_metrics["actor/global_block_cos"] = None
                mini_batch_metrics["actor/global_block_ang_deg"] = None

            if len(precond_block_cos_list) > 0:
                gbp = float(sum(precond_block_cos_list) / len(precond_block_cos_list))
                mini_batch_metrics["actor/global_block_cos_precond"] = gbp
                mini_batch_metrics["actor/global_block_ang_precond_deg"] = _ang_from_cos(gbp)
            else:
                mini_batch_metrics["actor/global_block_cos_precond"] = None
                mini_batch_metrics["actor/global_block_ang_precond_deg"] = None

        # 彻底释放 CPU 端大块内存
        del grads_sft, grads_pg
        return mini_batch_metrics

    def pretty_fsdp_flat_name(self, n: str) -> str:
        # n 是你现在拿到的参数名
        if n.endswith("._flat_param"):
            if ".model.layers." in n:
                # 逐层
                idx = n.split(".model.layers.")[1].split(".")[0]
                return f"block_{idx}"
            else:
                # 顶层未被细分的那部分
                return "top_level_misc"  # 可能包含 embeddings/final_norm/lm_head 等
        return n

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # print(f' data meta info: {data.meta_info}')
        need_analyze_sft_grads = data.meta_info.get('need_analyze_sft_grads', False)
        print(f'need analyze sft gradients: {need_analyze_sft_grads}')
        need_analyze_off_grads = data.meta_info.get('need_analyze_off_grads', False)
        print(f'need analyze off policy gradients: {need_analyze_off_grads}')
        step_index = data.meta_info.get('step_index', 0)
        print(f'step index: {step_index}')
        contain_off_policy = data.meta_info.get('contain_off_policy', False)
        calculate_sft_loss = data.meta_info.get('calculate_sft_loss', None)
        if calculate_sft_loss is None:
            calculate_sft_loss = self.config.calculate_sft_loss
        print(f'calculate_sft_loss: {calculate_sft_loss}')
        calculate_rl_loss = data.meta_info.get('calculate_rl_loss', None)
        if calculate_rl_loss is None:
            calculate_rl_loss = self.config.calculate_rl_loss
        print(f'calculate_rl_loss: {calculate_rl_loss}')
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "tgt_input_ids",
            "tgt_attention_mask",
            "tgt_responses"
        ]
        if contain_off_policy:
            select_keys.append('on_policy_mask')
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        select_data = data.select(batch_keys=select_keys)
        batch = select_data.batch
        non_tensor_batch = select_data.non_tensor_batch
        # print(f' non tensors batch keys: {non_tensor_batch.keys()}')
        # for k, v in batch.items():
        #     print(f'{k}: {v.shape}')
        # print(non_tensor_batch['uid'])

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        # print('ppo mini batch: ', self.config.ppo_mini_batch_size)
        clip_ratio = self.config.clip_ratio
        clip_ratio_low = (
            self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        )
        clip_ratio_high = (
            self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        )
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
        metrics = {}
        # print(f"初始显存: {torch.cuda.memory_allocated(device=get_device_id()) / 1024**3:.2f} GB")
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # print(f' data: {data}')
                responses_length = data[0]["responses"].shape[-1]
                prompt_length = data[0]["input_ids"].shape[-1] - responses_length

                if need_analyze_sft_grads:
                    out_dir = self.config.get("analyze_dump_dir", "./grad_dumps/on_pg_sft")
                    sft_batch_data = data[0:1].clone()
                    sft_batch_data = TensorDict(
                        {'input_ids': sft_batch_data["tgt_input_ids"],
                         'attention_mask': sft_batch_data["tgt_attention_mask"],
                         'position_ids': sft_batch_data["position_ids"],
                         'responses': sft_batch_data["tgt_responses"], }
                    )
                    sft_batch_data['response_mask'] = sft_batch_data['attention_mask'][:, prompt_length:]
                    sft_manifest = self._grab_grads_recompute("sft", sft_batch_data, sft_batch_data['response_mask'],
                            temperature=temperature, dp_no_sync=False, return_grads=False, save_dir=out_dir,
                            save_cpu_dtype=torch.float16, per_param_files=True, save_optimizer_state=False,
                            step=step_index, mini_batch_idx=batch_idx)

                    rl_batch_data = data[0:1].clone()
                    rl_batch_data = TensorDict(
                        {'input_ids': rl_batch_data["input_ids"],
                         'attention_mask': rl_batch_data["attention_mask"],
                         'position_ids': rl_batch_data["position_ids"],
                         'responses': rl_batch_data["responses"],
                         'old_log_probs': rl_batch_data["old_log_probs"],
                         'advantages': rl_batch_data["advantages"], }
                    )
                    rl_batch_data['response_mask'] = rl_batch_data['attention_mask'][:, prompt_length:]
                    pg_manifest = self._grab_grads_recompute("pg", rl_batch_data, rl_batch_data['response_mask'],
                            temperature, dp_no_sync=False, clip_ratio=clip_ratio, clip_ratio_high=clip_ratio_high,
                            clip_ratio_low=clip_ratio_low, clip_ratio_c=clip_ratio_c, return_grads=False,
                            save_dir=out_dir, save_cpu_dtype=torch.float16, per_param_files=True,
                            save_optimizer_state=False, step=step_index, mini_batch_idx=batch_idx)

                    # mini_batch_metrics = self.analyze_grads(grads_sft, grads_pg)
                    del sft_batch_data, rl_batch_data

                if need_analyze_off_grads:
                    out_dir = self.config.get("analyze_dump_dir", "./grad_dumps/off_on_pg")
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    test_data = data.clone()
                    on_pg_batch_data, off_pg_batch_data = split_on_policy_batch(test_data)
                    on_pg_batch_data = TensorDict(
                        {'input_ids': on_pg_batch_data["input_ids"],
                         'attention_mask': on_pg_batch_data["attention_mask"],
                         'position_ids': on_pg_batch_data["position_ids"],
                         'responses': on_pg_batch_data["responses"],
                         'old_log_probs': on_pg_batch_data["old_log_probs"],
                         'advantages': on_pg_batch_data["advantages"],
                        },
                        batch_size=on_pg_batch_data.batch_size,
                    )
                    # print(f' on policy data: {on_pg_batch_data}')
                    on_pg_batch_data['response_mask'] = on_pg_batch_data['attention_mask'][:, prompt_length:]
                    on_pg_batch_data_splits, _ = rearrange_micro_batches(on_pg_batch_data, max_token_len)
                    # print(f' on batch splits: {len(on_pg_batch_data_splits)}')
                    for t, on_pg_batch_data_split in enumerate(on_pg_batch_data_splits):
                        on_pg_manifest = self._grab_grads_recompute("on_pg", on_pg_batch_data_split,
                                on_pg_batch_data_split['response_mask'], temperature, dp_no_sync=False,
                                clip_ratio=clip_ratio, clip_ratio_high=clip_ratio_high, clip_ratio_low=clip_ratio_low,
                                clip_ratio_c=clip_ratio_c, return_grads=False, save_dir=out_dir,
                                save_cpu_dtype=torch.float16, per_param_files=True, save_optimizer_state=False,
                                step=step_index, mini_batch_idx=t*100+batch_idx,
                                on_policy_mask=torch.ones_like(on_pg_batch_data_split['response_mask']))
                    off_pg_batch_data = TensorDict(
                        {'input_ids': off_pg_batch_data["input_ids"],
                         'attention_mask': off_pg_batch_data["attention_mask"],
                         'position_ids': off_pg_batch_data["position_ids"],
                         'responses': off_pg_batch_data["responses"],
                         'old_log_probs': off_pg_batch_data["old_log_probs"],
                         'advantages': off_pg_batch_data["advantages"],
                        },
                        batch_size=off_pg_batch_data.batch_size,
                    )
                    off_pg_batch_data['response_mask'] = off_pg_batch_data['attention_mask'][:, prompt_length:]
                    # print(f' off policy data: {off_pg_batch_data}')
                    off_pg_batch_data_splits, _ = rearrange_micro_batches(off_pg_batch_data, max_token_len)
                    # print(f' off batch splits: {len(off_pg_batch_data_splits)}')
                    for t, off_pg_batch_data_split in enumerate(off_pg_batch_data_splits):
                        off_pg_manifest = self._grab_grads_recompute("off_pg", off_pg_batch_data_split,
                                off_pg_batch_data_split['response_mask'], temperature, dp_no_sync=False,
                                clip_ratio=clip_ratio, clip_ratio_high=clip_ratio_high, clip_ratio_low=clip_ratio_low,
                                clip_ratio_c=clip_ratio_c, return_grads=False, save_dir=out_dir,
                                save_cpu_dtype=torch.float16, per_param_files=True, save_optimizer_state=False,
                                step=step_index, mini_batch_idx=t * 100 + batch_idx,
                                on_policy_mask=torch.zeros_like(off_pg_batch_data_split['response_mask']))

                # split batch into micro_batches
                mini_batch = data
                # print('mini batch: ', mini_batch)
                if self.config.use_dynamic_bsz:
                    # print('ppo_max_token_len_per_gpu: ', self.config.ppo_max_token_len_per_gpu)
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches_with_targets(
                        batch=mini_batch, max_token_len=max_token_len,
                        with_sft=calculate_sft_loss, with_rl=calculate_rl_loss)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                micro_batches = [convert_data(batch) for batch in micro_batches]

                mini_batch_metrics = {}

                for index, data in enumerate(micro_batches):
                    micro_batch_metrics = {}
                    # print(f'micro batch data: {data}')

                    if calculate_sft_loss and calculate_rl_loss:
                        forward_batch_data = TensorDict(
                            {'input_ids': torch.cat([data["input_ids"], data["tgt_input_ids"]], dim=0),
                            'attention_mask': torch.cat([data["attention_mask"], data["tgt_attention_mask"]], dim=0),
                            'position_ids': torch.cat([data["position_ids"], data["position_ids"]], dim=0),
                            'responses': torch.cat([data["responses"], data["tgt_responses"]], dim=0),
                            'old_log_probs': torch.cat([data["old_log_probs"], data["old_log_probs"]], dim=0),
                            'advantages': torch.cat([data["advantages"], data["advantages"]], dim=0),},
                            batch_size=data.batch_size[0]*2,
                        )
                        forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        forward_batch_data['policy_mask'] = torch.cat(
                            [torch.ones_like(data["response_mask"]), torch.zeros_like(data["response_mask"])], dim=0).bool()
                        old_log_prob = forward_batch_data["old_log_probs"]
                        advantages = forward_batch_data["advantages"]
                    elif not calculate_sft_loss and calculate_rl_loss:
                        forward_batch_data = TensorDict(
                            {'input_ids': data["input_ids"],
                            'attention_mask': data["attention_mask"],
                            'position_ids': data["position_ids"],
                            'responses': data["responses"],
                            'old_log_probs': data["old_log_probs"],
                            'advantages': data["advantages"],},
                            batch_size=data.batch_size,
                        )
                        forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        forward_batch_data['policy_mask'] = torch.ones_like(data["response_mask"]).bool()
                        if contain_off_policy:
                            forward_batch_data['on_policy_mask'] = data["on_policy_mask"]
                            # print(f'forward data {forward_batch_data}')
                            forward_batch_data, _ = split_on_policy_batch(forward_batch_data)
                            if forward_batch_data is None:
                                continue
                            # print(f'filtered on policy batch {forward_batch_data}')
                        old_log_prob = forward_batch_data["old_log_probs"]
                        advantages = forward_batch_data["advantages"]
                    elif calculate_sft_loss and not calculate_rl_loss:
                        # 这个分支目前不会跑到
                        forward_batch_data = TensorDict(
                            {'input_ids': data["tgt_input_ids"],
                            'attention_mask': data["tgt_attention_mask"],
                            'position_ids': data["position_ids"],
                            'responses': data["tgt_responses"],},
                            batch_size=data.batch_size,
                        )
                        forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        forward_batch_data['policy_mask'] = torch.zeros_like(data["tgt_responses"]).bool()
                    else:
                        raise ValueError('both sft loss and rl loss are not calculated')
                    response_mask = forward_batch_data["response_mask"]

                    # print(f' forward batch data: {forward_batch_data}')
                    # all return: (bsz, response_length)
                    entropy, all_log_prob = self._forward_micro_batch(
                        micro_batch=forward_batch_data, temperature=temperature
                    )

                    if calculate_sft_loss:
                        policy_mask = forward_batch_data['policy_mask']
                        sft_loss = compute_sft_loss(all_log_prob[~policy_mask], response_mask[~policy_mask])
                        # print('sft loss: ', sft_loss)
                    else:
                        sft_loss = torch.tensor(0.0, device=get_device_id())

                    if calculate_rl_loss:
                        policy_mask = forward_batch_data['policy_mask']
                        pg_loss, on_pg_loss, on_pg_clipfrac, off_pg_loss, ppo_kl = compute_token_mixed_policy_loss(
                            old_log_prob=old_log_prob[policy_mask],
                            log_prob=all_log_prob[policy_mask],
                            advantages=advantages[policy_mask],
                            eos_mask=response_mask[policy_mask],
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        # print('pg loss: ', pg_loss, ' pg_loss.shape: ', pg_loss.shape)
                        data = {
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/on_pg_clipfrac': on_pg_clipfrac.detach().item(),
                        }
                        append_to_dict(metrics, data)
                    else:
                        pg_loss = torch.tensor(0.0, device=get_device_id())
                        print('not calculate rl loss')

                    # TODO: 看看适应性温度的影响，可以参考：/home/hzchen/jyh/LUFFY-main/luffy/verl/verl/mix_src/mix_actor.py
                    # 中的 205 行开始的内容

                    if calculate_sft_loss and calculate_rl_loss:
                        if self.config.sft_loss_coef < 1e-4:
                            all_loss = pg_loss
                        else:
                            all_loss = pg_loss + sft_loss * self.config.sft_loss_coef
                    elif calculate_sft_loss and not calculate_rl_loss:
                        all_loss = sft_loss * self.config.sft_loss_coef
                    elif not calculate_sft_loss and calculate_rl_loss:
                        all_loss = pg_loss
                    else:
                        raise ValueError('both sft loss and rl loss are not calculated')

                    # print(f'all loss: {all_loss}')
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = all_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = all_loss / self.gradient_accumulation
                    loss.backward()

                    if calculate_sft_loss:
                        micro_batch_metrics["actor/sft_loss"] = sft_loss.detach().item()
                        micro_batch_metrics["actor/sft_coef"] = self.config.sft_loss_coef

                    if calculate_rl_loss:
                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics["actor/grad_norm"] = grad_norm.detach().item()
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
