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
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import ceildiv, get_seqlen_balanced_partitions, roundup_divisible, \
    rearrange_micro_batches, prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

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

def compute_sft_loss(log_prob, mask):
    assert log_prob.shape == mask.shape, f'log_prob shape {log_prob.shape} does not match mask shape {mask.shape}'
    valid_tokens = torch.sum(mask).item()
    print(f"SFT有效token数量: {valid_tokens}")
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
                     log_prob: Tensor,
                     policy_reshape_weight: float = 1.0,
                     policy_reshape_pow_exp: float = 0.5,
                     eps: float = 1e-8):
    # 这里不是严格 ratio，更多是“提升该 token 概率”的系数
    off_ratio = torch.exp(log_prob)  # (0, 1]
    off_frac_min_before = torch.min(off_ratio, dim=-1).values
    off_frac_max_before = torch.max(off_ratio, dim=-1).values
    off_frac_mean_before = torch.mean(off_ratio)
    off_frac_std_before = torch.std(off_ratio)
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
    off_frac_min_after = torch.min(off_ratio, dim=-1).values
    off_frac_max_after = torch.max(off_ratio, dim=-1).values
    off_frac_mean_after = torch.mean(off_ratio, dim=-1)
    off_frac_std_after = torch.std(off_ratio, dim=-1)
    stats = {
        "off_frac_min_before": off_frac_min_before.mean(),
        "off_frac_max_before": off_frac_max_before.mean(),
        "off_frac_mean_before": off_frac_mean_before.mean(),
        "off_frac_std_before": off_frac_std_before.mean(),
        "off_frac_min_after": off_frac_min_after.mean(),
        "off_frac_max_after": off_frac_max_after.mean(),
        "off_frac_mean_after": off_frac_mean_after.mean(),
        "off_frac_std_after": off_frac_std_after.mean(),
    }
    return off_ratio, stats  # ← 必须返回

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
    loss_agg_mode='token-mean',
    loss_remove_clip=False,
):
    # 1) 形状与基本检查
    assert log_prob.shape == old_log_prob.shape, f'log_prob shape {log_prob.shape} != old_log_prob shape {old_log_prob.shape}'
    assert log_prob.shape == advantages.shape,   f'log_prob shape {log_prob.shape} != advantages shape {advantages.shape}'
    assert log_prob.shape == eos_mask.shape,     f'log_prob shape {log_prob.shape} != eos_mask shape {eos_mask.shape}'

    # print(f'shape of log_prob {log_prob.shape}')
    # 统一设备/类型的零
    valid_tokens = torch.sum(eos_mask).item()
    print(f"RL有效token数量: {valid_tokens}")
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
        off_ratio, stats = shape_off_policy(
            off_policy_reshape, log_prob,
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
        stats = None

    # 5) 总 loss 的归一化
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    stats["off_pg_loss"] = off_pg_loss
    return pg_loss, on_pg_loss, on_pg_clipfrac, ppo_kl, stats

def prepare_dynamic_batch_with_targets(data: DataProto, max_token_len: int,
                                       with_sft: bool=False, with_rl: bool=True) -> tuple[list[DataProto], list[list[int]]]:
    """
    Prepare a batch for dynamic batching.

    Args:
        data (DataProto): The input data.
        max_token_len (int): The maximum token length for dynamic batching.
        with_sft:
        with_rl:

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
    """
    batch, batch_idx_list = rearrange_micro_batches_with_targets(data.batch, max_token_len=max_token_len,
                                                                 with_sft=with_sft, with_rl=with_rl)
    micro_batches = []
    for i, batch_idx in enumerate(batch_idx_list):
        tensors = dict(batch[i])
        non_tensors = {key: value[batch_idx] for key, value in data.non_tensor_batch.items()}
        micro_batches.append(DataProto.from_dict(tensors, non_tensors))

    return micro_batches, batch_idx_list

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

def _sanitize_name(name: str) -> str:
    # 便于作为文件名（尽量短）
    base = name.replace("/", "_").replace(".", "_").replace(" ", "_")
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}"


class MixedTrainParallelPPOActor(DataParallelPPOActor):

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

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, need_eos_prob=False, eos_token_id=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy:     (bs, response_len)  or None
            log_probs:   (bs, response_len)
            eos_prob:    (bs, response_len)  新增：下一 token 是 eos 的概率
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        if need_eos_prob:
            assert eos_token_id is not None, "eos_token_id must be provided when need_eos_prob=True"
            # 临时关闭 fused kernel，因为我们必须拿到完整 logits
            orig_use_fused = self.use_fused_kernels # TODO: 临时关闭 fused kernel
            self.use_fused_kernels = False
        else:
            orig_use_fused = None

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    if need_eos_prob:
                        probs_rmpad = torch.softmax(logits_rmpad, dim=-1)  # (total_nnz, vocab)
                        eos_prob_rmpad = probs_rmpad[:, eos_token_id]  # (total_nnz,)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ).squeeze(-1)
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)
                if need_eos_prob:
                    full_eos_prob = pad_input(eos_prob_rmpad.unsqueeze(-1), indices, batch_size, seqlen).squeeze(-1)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs[:, -response_length - 1 : -1]  # (bsz, response_length)
                if need_eos_prob:
                    eos_prob = full_eos_prob[:, -response_length - 1: -1]
                else:
                    eos_prob = None

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if need_eos_prob:
                        probs = torch.softmax(logits, dim=-1)  # (bs, seqlen, vocab)
                        eos_prob = probs[:, :, eos_token_id]  # (bs, seqlen)
                    else:
                        eos_prob = None
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if need_eos_prob:
                self.use_fused_kernels = orig_use_fused

            return entropy, log_probs, eos_prob

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, need_eos_prob=False, eos_token_id=-1) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        eos_prob_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, eos_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                    need_eos_prob=need_eos_prob, eos_token_id=eos_token_id
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if need_eos_prob:
                eos_prob_lst.append(eos_prob)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        eos_probs = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if need_eos_prob:
            eos_probs = torch.concat(eos_prob_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if need_eos_prob:
                eos_probs = restore_dynamic_batch(eos_probs, batch_idx_list)

        return log_probs, entropys, eos_probs

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
            "tgt_responses",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if contain_off_policy:
            select_keys.append('on_policy_mask')
            select_data = data.select(batch_keys=select_keys)
        else:
            select_data = data.select(batch_keys=select_keys)
            select_data.batch["on_policy_mask"] = torch.ones_like(select_data.batch["response_mask"])
        non_tensor_batch = select_data.non_tensor_batch
        # print(f' non tensors batch keys: {non_tensor_batch.keys()}')
        # for k, v in batch.items():
        #     print(f'{k}: {v.shape}')
        # print(non_tensor_batch['uid'])

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if calculate_rl_loss and calculate_sft_loss:
            mini_batches = [select_data]
        else:
            print(f'ppo mini batch size: {self.config.ppo_mini_batch_size}')
            mini_batches = select_data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        if on_policy:
            print('num of mini batches is 1')
        else:
            print('num of mini batches: ', len(mini_batches))

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
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # print(f' data: {data}')
                responses_length = mini_batch.batch["responses"].shape[-1]
                prompt_length = mini_batch.batch["input_ids"].shape[-1] - responses_length

                # TODO: 重新看看这一部分分析SFT和RL梯度的内容
                if need_analyze_sft_grads:
                    out_dir = self.config.get("analyze_dump_dir", "./grad_dumps/on_pg_sft")
                    sft_batch_data = mini_batch[0:1].clone()
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

                    rl_batch_data = mini_batch[0:1].clone()
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

                # TODO: 重新看看这一部分分析online RL和offline RL梯度的内容
                if need_analyze_off_grads:
                    out_dir = self.config.get("analyze_dump_dir", "./grad_dumps/off_on_pg")
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    test_data = mini_batch.clone()
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
                # print('mini batch: ', mini_batch)
                if self.config.use_dynamic_bsz:
                    # print('ppo_max_token_len_per_gpu: ', self.config.ppo_max_token_len_per_gpu)
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch_with_targets(
                        data=mini_batch, max_token_len=max_token_len,
                        with_sft=calculate_sft_loss, with_rl=calculate_rl_loss)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                mini_batch_metrics = {}

                for index, micro_batch in enumerate(micro_batches):
                    micro_batch_metrics = {}
                    # print(f'micro batch data: {data}')
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    batch_size = response_mask.shape[0]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    if calculate_sft_loss and calculate_rl_loss:
                        forward_batch_data = TensorDict(
                            {'input_ids': torch.cat([model_inputs["input_ids"], model_inputs["tgt_input_ids"]], dim=0),
                            'attention_mask': torch.cat([model_inputs["attention_mask"], model_inputs["tgt_attention_mask"]], dim=0),
                            'position_ids': torch.cat([model_inputs["position_ids"], model_inputs["position_ids"]], dim=0),
                            'responses': torch.cat([model_inputs["responses"], model_inputs["tgt_responses"]], dim=0),
                            'old_log_probs': torch.cat([model_inputs["old_log_probs"], model_inputs["old_log_probs"]], dim=0),
                            'advantages': torch.cat([model_inputs["advantages"], model_inputs["advantages"]], dim=0),},
                            batch_size=batch_size*2,
                        )
                        forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        forward_batch_data['policy_mask'] = torch.cat(
                            [torch.ones_like(model_inputs["response_mask"]), torch.zeros_like(model_inputs["response_mask"])], dim=0).bool()
                        forward_batch_data['on_policy_mask'] = torch.cat(
                            [torch.ones_like(model_inputs["on_policy_mask"]), torch.zeros_like(model_inputs["response_mask"])], dim=0).bool()
                        advantages = forward_batch_data["advantages"]
                    elif not calculate_sft_loss and calculate_rl_loss:
                        # print('only rl loss')
                        forward_batch_data = TensorDict(
                            {'input_ids': model_inputs["input_ids"],
                            'attention_mask': model_inputs["attention_mask"],
                            'position_ids': model_inputs["position_ids"],
                            'responses': model_inputs["responses"],
                            'old_log_probs': model_inputs["old_log_probs"],
                            'advantages': model_inputs["advantages"],
                            'on_policy_mask': model_inputs["on_policy_mask"],},
                            batch_size=batch_size,
                        )
                        forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        forward_batch_data['policy_mask'] = torch.ones_like(model_inputs["response_mask"]).bool()
                        advantages = forward_batch_data["advantages"]
                    elif calculate_sft_loss and not calculate_rl_loss:
                        raise NotImplementedError
                        # # 这个分支目前不会跑到
                        # forward_batch_data = TensorDict(
                        #     {'input_ids': data["tgt_input_ids"],
                        #     'attention_mask': data["tgt_attention_mask"],
                        #     'position_ids': data["position_ids"],
                        #     'responses': data["tgt_responses"],},
                        #     batch_size=data.batch_size,
                        # )
                        # forward_batch_data['response_mask'] = forward_batch_data['attention_mask'][:, prompt_length:]
                        # forward_batch_data['policy_mask'] = torch.zeros_like(data["tgt_responses"]).bool()
                    else:
                        raise ValueError('both sft loss and rl loss are not calculated')

                    response_mask = forward_batch_data["response_mask"]
                    on_policy_mask = forward_batch_data["on_policy_mask"]
                    # all return: (bsz, response_length)

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # print(f' forward batch data: {forward_batch_data}')
                    # all return: (bsz, response_length)
                    entropy, all_log_prob, _ = self._forward_micro_batch(
                        micro_batch=forward_batch_data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = all_log_prob.detach()
                    else:
                        old_log_prob = forward_batch_data["old_log_probs"]

                    if calculate_sft_loss:
                        policy_mask = forward_batch_data['policy_mask']
                        sft_loss = compute_sft_loss(all_log_prob*(~policy_mask), response_mask*(~policy_mask))
                        # print('sft loss: ', sft_loss)
                    else:
                        sft_loss = torch.tensor(0.0, device=get_device_id())

                    if calculate_rl_loss:
                        policy_mask = forward_batch_data['policy_mask']
                        pg_loss, on_pg_loss, on_pg_clipfrac, ppo_kl, stats = compute_token_mixed_policy_loss(
                            old_log_prob=old_log_prob*policy_mask,
                            log_prob=all_log_prob*policy_mask,
                            advantages=advantages*policy_mask,
                            eos_mask=response_mask*policy_mask,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            on_policy_mask=on_policy_mask*policy_mask,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            loss_agg_mode=loss_agg_mode,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        # print('on pg loss: ', on_pg_loss, ' off pg loss: ', off_pg_loss)
                        data = {
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/on_pg_clipfrac': on_pg_clipfrac.detach().item(),
                            'actor/ppo_kl': ppo_kl.detach().item(),
                            'actor/off_pg_loss': stats['off_pg_loss'].detach().item(),
                            'actor/off_frac_min_before': stats['off_frac_min_before'].detach().item(),
                            'actor/off_frac_min_after': stats['off_frac_min_after'].detach().item(),
                            'actor/off_frac_max_before': stats['off_frac_max_before'].detach().item(),
                            'actor/off_frac_max_after': stats['off_frac_max_after'].detach().item(),
                            'actor/off_frac_mean_before': stats['off_frac_mean_before'].detach().item(),
                            'actor/off_frac_mean_after': stats['off_frac_mean_after'].detach().item(),
                            'actor/off_frac_std_before': stats['off_frac_std_before'].detach().item(),
                            'actor/off_frac_std_after': stats['off_frac_std_after'].detach().item(),
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

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        all_loss = all_loss - entropy_loss * entropy_coeff

                    # print(f'all loss: {all_loss}')
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = all_loss * loss_scale_factor
                    else:
                        loss = all_loss * loss_scale_factor
                    loss.backward()

                    if calculate_sft_loss:
                        micro_batch_metrics["actor/sft_loss"] = sft_loss.detach().item()
                        micro_batch_metrics["actor/sft_coef"] = self.config.sft_loss_coef

                    if calculate_rl_loss:
                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics["actor/grad_norm"] = grad_norm.detach().item()
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
