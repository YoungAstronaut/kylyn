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

import itertools
import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.dp_actor import DataParallelPPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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

def compute_token_mixed_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    cliprange_low,
    cliprange_high,
    clip_ration_c,
    prefix_mask, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    # TODO: 补充这一段的注释，可以参考默认的PPOActor
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # TODO: 测试on-policy是否需要shape
    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }


class MixedTrainParallelPPOActor(DataParallelPPOActor):

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        print('data used for training: ', data.batch)
        print('keys of data batch: ', data.batch.keys())
        print('keys of data non_tensor_batch: ', data.non_tensor_batch.keys())
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "tgt_input_ids",
            "prefix_mask", # 用于混合策略的强化学习
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        print('selected batch: ', batch)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        print('ppo mini batch: ', self.config.ppo_mini_batch_size)
        print('length of mini batch: ', len(dataloader))
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                print(type(mini_batch))
                print('mini batch: ', mini_batch)
                if self.config.use_dynamic_bsz:
                    print('ppo_max_token_len_per_gpu: ', self.config.ppo_max_token_len_per_gpu)
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # TODO: 调试用，后面应该移除
                # print('before split: ', mini_batch)
                # micro_batches = mini_batch.split(4, dim=0)
                # print('after split: ', micro_batches)
                self.actor_optimizer.zero_grad()

                print('length of micro batches: ', len(micro_batches))
                for data in micro_batches:
                    micro_batch_metrics = {}

                    print('data: ', data)
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    print('data type: ', type(data))
                    print('processed data: ', data)
                    # TODO: 将data翻倍，前一半是rl，后一半是sft。input_ids, attention_mask, position_ids, responses这四个成员需要翻倍
                    responses_length = data["responses"].shape[1]
                    prompt_length = data["input_ids"].shape[1] - responses_length
                    tgt_inputs_length = data["tgt_input_ids"].shape[1]
                    batch_length = data["input_ids"].shape[0]
                    # 获取pad_token_id
                    pad_positions = (data['attention_mask'] == 0)
                    pad_token_id = data['input_ids'][pad_positions].unique().item()
                    target_pad = torch.full((batch_length, responses_length-tgt_inputs_length), pad_token_id).to(get_device_id())
                    sft_responses = torch.cat([data['tgt_input_ids'], target_pad], dim=-1)
                    sft_input_ids = torch.cat(
                        [data["input_ids"][:, :prompt_length], sft_responses], 
                        dim=-1
                    )
                    attention_pad = torch.full((batch_length, responses_length-tgt_inputs_length), 0).to(get_device_id())
                    sft_attention_mask = torch.cat(
                        [data["attention_mask"][:, :prompt_length], (data['tgt_input_ids'] != pad_token_id).int(), attention_pad], 
                        dim=-1
                    )
                    position_start_index = data["position_ids"][:, -1] + 1
                    position_start_index = position_start_index.unsqueeze(-1)
                    new_positions = position_start_index + torch.arange(0, responses_length, device=get_device_id())
                    sft_position_ids = torch.cat(
                        [data["position_ids"][:, :prompt_length], new_positions], 
                        dim=-1
                    )
                    print('sft input ids: ', sft_input_ids)
                    print('sft attention mask: ', sft_attention_mask)
                    print('sft position ids: ', sft_position_ids)
                    print('sft responses: ', sft_responses)
                    if self.config.calculate_sft_loss:
                        forward_batch_data = TensorDict(
                            {'input_ids': torch.cat([data["input_ids"], sft_input_ids], dim=0),
                            'attention_mask': torch.cat([data["attention_mask"], sft_attention_mask], dim=0),
                            'position_ids': torch.cat([data["position_ids"], sft_position_ids], dim=0),
                            'responses': torch.cat([data["responses"], sft_responses], dim=0),}
                        )
                        print('input data: ', data)
                        print('forward data: ', forward_batch_data)
                    else:
                        forward_batch_data = TensorDict(
                            {'input_ids': data["input_ids"],
                            'attention_mask': data["attention_mask"],
                            'position_ids': data["position_ids"],
                            'responses': data["responses"],}
                        )
                        print('forward data: ', forward_batch_data)
                    
                    response_mask = data["response_mask"]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, all_log_prob = self._forward_micro_batch(
                        micro_batch=forward_batch_data, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    print('forward results ************ ')
                    print('entropy: ', entropy.shape if entropy is not None else None)
                    print('all log prob: ', all_log_prob.shape)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.calculate_sft_loss:
                        log_prob = all_log_prob[batch_length:, ...]
                        print('log prob for sft', log_prob.shape)
                        valid_log_prob = response_mask * log_prob
                        sft_loss = -torch.sum(valid_log_prob, dim=-1) / torch.sum(response_mask, dim=-1)
                        sft_loss = torch.mean(sft_loss)
                        sft_loss.backward()
                        sft_grads = {name: param.grad.detach().cpu().clone() for name, param in self.actor_module.named_parameters()}  
                        for name, grad in self.actor_module.named_parameters():
                            print('grad before zero: ', name, ' : ', torch.norm(grad, p=2).item())
                        torch.cuda.empty_cache()
                        self.actor_optimizer.zero_grad()  # 清空梯度  
 
                    else:
                        sft_loss = torch.tensor(0.0, device=get_device_id())
                        sft_grads = None

                    if loss_mode == "vanilla":
                        log_prob = all_log_prob[:batch_length, ...]
                        print('log prob for rl', log_prob)
                        ret_dict = compute_token_mixed_policy_loss(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ration_c=clip_ratio_c,
                            prefix_mask=data['prefix_mask'],
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']

                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)
                        # 计算RL Loss梯度  
                        pg_loss.backward(retain_graph=True)
                        rl_grads = {name: param.grad.detach().cpu().clone() for name, param in self.actor_module.named_parameters()}
                        torch.cuda.empty_cache()
                        self.actor_optimizer.zero_grad()  # 清空梯度

                    else:
                        raise NotImplementedError(f"loss_mode {loss_mode} not implemented")


                    if sft_grads and rl_grads:
                        gradient_analysis = analyze_gradients(sft_grads, rl_grads)

                    # TODO: 看看适应性温度的影响，可以参考：/home/hzchen/jyh/LUFFY-main/luffy/verl/verl/mix_src/mix_actor.py
                    # 中的 205 行开始的内容
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.calculate_sft_loss:
                        policy_loss = policy_loss + sft_loss * self.config.sft_loss_coef
                        micro_batch_metrics["actor/sft_loss"] = sft_loss.detach().item()
                        micro_batch_metrics["actor/sft_coef"] = self.config.sft_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
