"""
Single Process Actor
"""
import logging
import os
from typing import Optional

from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input

import torch
from torch import distributed as dist
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import ceildiv, get_seqlen_balanced_partitions, roundup_divisible, \
    prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.config.actor import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EPS = 1e-12

def gather_tgt_inputs(data) -> TensorDict | None:
    tgt_rm = data["tgt_response_mask"]  # (B, resp_len)

    # 更便宜：any 比 sum 更轻
    valid = tgt_rm.bool().any(dim=-1)   # (B,) bool

    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None

    def take(x):  # x: (B, ...)
        return x.index_select(0, idx)

    out = {
        "input_ids":      take(data["tgt_input_ids"]),
        "attention_mask": take(data["tgt_attention_mask"]),
        "position_ids":   take(data["tgt_position_ids"]),
        "responses":      take(data["tgt_responses"]),
        "response_mask":  take(data["tgt_response_mask"]),
    }

    # 更简洁：zeros_like
    out["old_log_probs"] = torch.zeros_like(out["response_mask"], dtype=data["old_log_probs"].dtype)

    return TensorDict(out, batch_size=out["input_ids"].shape[0])

import torch

def compute_dft_loss(
    log_prob: torch.Tensor,
    mask: torch.Tensor,
    loss_agg_mode: str,
    dft_alpha: float = 0.5,       # 0 -> SFT, 1 -> DFT
    eps: float = 1e-8,
    normalize: bool = True,       # 保持尺度更稳定（建议 True）
    debug: bool = False,
):
    """
    log_prob: log p(y*_t | ...) for target tokens, shape (B, T)
    mask:     0/1 mask, shape (B, T)
    """
    assert loss_agg_mode == "token-mean"
    assert log_prob.shape == mask.shape

    # token prob p_t = exp(log p_t); 用 float32 做 exp 更稳
    p = torch.exp(log_prob.float()).clamp_min(eps).detach()  # sg(p)
    if dft_alpha != 1.0:
        # 平滑插值：α=0 完全不加权（SFT），α=1 完全 DFT
        p = p.pow(dft_alpha)

    p = p.to(log_prob.dtype)

    # DFT: - sg(p) * log p
    loss_mat = -(p * log_prob)

    # 可选：归一化，使得整体 loss 尺度不因为平均 p 变小而整体变小
    # 这对你“RL + SFT 混合”很关键，否则 SFT 梯度可能突然变弱/变强
    if normalize:
        denom = mask.sum().clamp_min(1).to(p.dtype)
        p_mean = (p * mask).sum() / denom
        loss_mat = loss_mat / p_mean.clamp_min(eps)

    # token-mean（分母仍然是 token 数）
    batch_num_tokens = mask.sum().detach().clamp_min(1)
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=mask,
        loss_agg_mode="token-mean",
        batch_num_tokens=batch_num_tokens,
    )

    # 若没有有效 token，loss 应该是 0（这里给个兜底）
    loss = loss * (batch_num_tokens > 0).to(loss.dtype)

    if debug:
        vt = int(batch_num_tokens.detach().cpu())
        print(f"[DFT-SFT] valid_tokens={vt}, p_mean={float(p_mean.detach().cpu()) if normalize else 'n/a'}")

    return loss

def compute_sft_loss(log_prob, mask, loss_agg_mode, debug=False):
    assert loss_agg_mode == "token-mean"
    assert log_prob.shape == mask.shape

    loss_mat = -log_prob

    den = mask.sum().detach()                 # scalar tensor
    den_safe = den.clamp_min(1)               # avoid div0

    # 关键：显式把 batch_num_tokens 传给 agg_loss，避免它内部用 0 当分母
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=mask,
        loss_agg_mode="token-mean",
        batch_num_tokens=den_safe,            # <-- 这里
        # dp_size 先不管单卡就默认 1
    )

    # 若 den==0，本应 loss=0（而不是 num/1），这里乘回去即可
    loss = loss * (den > 0).to(loss.dtype)

    if debug:
        valid_tokens = int(den.cpu())
        per_seq = mask.sum(dim=-1).detach().cpu().tolist()
        print(f"SFT有效token数量: {valid_tokens}, per_seq_tokens: {per_seq}")

    return loss

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
        td = batch[i]
        tensors = {k: td[k] for k in td.keys()}
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
        rl_len = batch["attention_mask"].sum(dim=1)
        tgt_valid = (batch["tgt_response_mask"].sum(dim=1) > 0)  # (B,) bool
        tgt_len = (batch["tgt_attention_mask"] * tgt_valid[:, None]).sum(dim=1)
        seq_len_effective = rl_len + tgt_len
    elif with_sft and not with_rl:
        tgt_valid = (batch["tgt_response_mask"].sum(dim=1) > 0)  # (B,)
        seq_len_effective = (batch["tgt_attention_mask"] * tgt_valid[:, None]).sum(dim=1)
    else:
        raise ValueError("with_sft and with_rl cannot be both False")
    total_seqlen = seq_len_effective.sum().item()
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


class MixedTrainParallelPPOActor(DataParallelPPOActor):
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
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:
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
        calculate_sft_loss = data.meta_info.get('calculate_sft_loss', None)
        if calculate_sft_loss is None:
            calculate_sft_loss = self.config.calculate_sft_loss
        # print(f"calculate_sft_loss: {calculate_sft_loss}")
        calculate_rl_loss = data.meta_info.get('calculate_rl_loss', None)
        if calculate_rl_loss is None:
            calculate_rl_loss = self.config.calculate_rl_loss
        # print(f"calculate_rl_loss: {calculate_rl_loss}")

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
            "tgt_response_mask",
            "tgt_position_ids"
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        select_data = data.select(batch_keys=select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # print(f'ppo mini batch size: {self.config.ppo_mini_batch_size}')
        mini_batches = select_data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        # if on_policy:
            # print('num of mini batches is 1')
        # else:
            # print('num of mini batches: ', len(mini_batches))

        metrics = {}
        # print(f"初始显存: {torch.cuda.memory_allocated(device=get_device_id()) / 1024**3:.2f} GB")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # responses_length = mini_batch.batch["responses"].shape[-1]
                # prompt_length = mini_batch.batch["input_ids"].shape[-1] - responses_length
                # split batch into micro_batches
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
                
                total_rl_tokens = mini_batch.batch["response_mask"].sum().detach()
                total_sft_tokens = mini_batch.batch["tgt_response_mask"].sum().detach()
                # print(f"total_rl_tokens: {total_rl_tokens}, total_sft_tokens: {total_sft_tokens}")
                total_rl_tokens = total_rl_tokens.clamp_min(1)
                total_sft_tokens = total_sft_tokens.clamp_min(1)
                # print(f"total_rl_tokens: {total_rl_tokens}, total_sft_tokens: {total_sft_tokens}")

                # print("micro batch length: ", len(micro_batches))
                for index, micro_batch in enumerate(micro_batches):
                    micro_batch_metrics = {}
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if calculate_sft_loss and calculate_rl_loss:
                        valid_tgt_input = gather_tgt_inputs(model_inputs)
                        # forward_batch_data = TensorDict(
                        #     {"input_ids": model_inputs["input_ids"],
                        #     "attention_mask": model_inputs["attention_mask"],
                        #     "position_ids": model_inputs["position_ids"],
                        #     "responses": model_inputs["responses"],
                        #     "response_mask": model_inputs["response_mask"],
                        #     "old_log_probs": model_inputs["old_log_probs"],},
                        #     batch_size=model_inputs["input_ids"].shape[0],
                        # )
                        RL_KEYS = ["input_ids","attention_mask","position_ids","responses","response_mask","old_log_probs"]
                        forward_batch_data = micro_batch.batch.select(*RL_KEYS)
                        # print("pre forward batch data: ", forward_batch_data)
                        # print("tgt forward batch data: ", valid_tgt_input)
                        if valid_tgt_input is not None:
                            forward_batch_data = TensorDict.cat([forward_batch_data, valid_tgt_input], dim=0)
                        #     print("forward batch data: ", forward_batch_data)
                        # else:
                        #     print("valid_tgt_input is None")
                        # for i in range(forward_batch_data.batch_size[0]):
                            # print("attention mask valid num: ", forward_batch_data["attention_mask"][i][prompt_length:].sum(),
                            #     "response position ids: ", forward_batch_data["position_ids"][i][prompt_length:],
                            #     "full response position ids: ", forward_batch_data["position_ids"][i][prompt_length:],
                            #       "response mask: ", forward_batch_data["response_mask"][i].sum(),)
                    elif not calculate_sft_loss and calculate_rl_loss:
                        forward_batch_data = TensorDict(
                            {"input_ids": model_inputs["input_ids"],
                            "attention_mask": model_inputs["attention_mask"],
                            "position_ids": model_inputs["position_ids"],
                            "responses": model_inputs["responses"],
                            "response_mask": model_inputs["response_mask"],
                            "old_log_probs": model_inputs["old_log_probs"],},
                            batch_size=model_inputs["input_ids"].shape[0],
                        )
                    elif calculate_sft_loss and not calculate_rl_loss:
                        forward_batch_data = gather_tgt_inputs(model_inputs)
                        if forward_batch_data is None:
                            continue  # 这个 micro 没有任何 tgt token，跳过
                    else:
                        raise ValueError('both sft loss and rl loss are not calculated')

                    calculate_entropy = (entropy_coeff != 0 and calculate_rl_loss)
                        
                    n_rl = model_inputs["input_ids"].shape[0] if calculate_rl_loss else 0

                    entropy, all_log_prob, _ = self._forward_micro_batch(
                        micro_batch=forward_batch_data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = all_log_prob.detach()
                    else:
                        old_log_prob = forward_batch_data["old_log_probs"]

                    if calculate_sft_loss:
                        logp_sft = all_log_prob[n_rl:]
                        mask_sft = forward_batch_data["response_mask"][n_rl:]
                        sft_loss = compute_dft_loss(logp_sft, mask_sft, loss_agg_mode)
                    else:
                        sft_loss = torch.zeros((), device=all_log_prob.device, dtype=all_log_prob.dtype)
                    # print("sft loss: ", sft_loss)

                    if calculate_rl_loss:
                        logp_rl = all_log_prob[:n_rl]
                        mask_rl = forward_batch_data["response_mask"][:n_rl]
                        rl_tokens_micro = mask_rl.sum().detach()

                        if rl_tokens_micro.item() == 0:   # rare branch，允许同步
                            pg_loss = torch.zeros((), device=all_log_prob.device, dtype=all_log_prob.dtype)
                            pg_metrics = {}
                            entropy_loss = torch.zeros_like(pg_loss)
                        else:
                            # print("mask_rl: ", mask_rl.sum().item())
                            old_rl = old_log_prob[:n_rl].detach()
                            adv_rl  = model_inputs["advantages"]
                            # print("advantages: ", adv_rl[..., 0])
                            # print(f"global batch info: {self.config.global_batch_info}")
                            pg_loss, pg_metrics = compute_policy_loss_vanilla(
                                old_log_prob=old_rl,
                                log_prob=logp_rl,
                                response_mask=mask_rl,
                                advantages=adv_rl,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )
                            if entropy_coeff != 0:
                                ent_rl = entropy[:n_rl]
                                # 也做 safe denom，彻底杜绝 0 分母
                                den = rl_tokens_micro.clamp_min(1)
                                entropy_loss = agg_loss(ent_rl, mask_rl, "token-mean", batch_num_tokens=den)
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        pg_loss = torch.zeros((), device=all_log_prob.device, dtype=all_log_prob.dtype)
                        # print('not calculate rl loss')
                    # print("pg loss: ", pg_loss)

                    # TODO: 看看适应性温度的影响，可以参考：/home/hzchen/jyh/LUFFY-main/luffy/verl/verl/mix_src/mix_actor.py
                    # 中的 205 行开始的内容
                    
                    if self.config.use_dynamic_bsz and loss_agg_mode == "token-mean":
                        # micro valid tokens
                        if calculate_rl_loss:
                            w_rl = (rl_tokens_micro / total_rl_tokens).to(all_log_prob.dtype)
                        else:
                            w_rl = 0.0

                        if calculate_sft_loss:
                            # forward_batch_data 的后半段是拼进去的 tgt（你把它的 response_mask 放进去了）
                            sft_tokens_micro = forward_batch_data["response_mask"][n_rl:].sum().detach()
                            # total_sft_tokens 可能为 0（比如某些 step 没有 tgt），这里你已经 clamp 过了
                            w_sft = (sft_tokens_micro / total_sft_tokens).to(all_log_prob.dtype)
                        else:
                            w_sft = 0.0
                        
                        # print("w_rl: ", w_rl, " w_sft: ", w_sft)

                        loss = w_rl * pg_loss + self.config.sft_loss_coef * w_sft * sft_loss
                        # print("pg loss: ", pg_loss, " sft loss: ", sft_loss, " loss: ", loss)

                        if entropy_coeff != 0 and calculate_rl_loss:
                            loss = loss - entropy_coeff * w_rl * entropy_loss
                    else:
                        # fallback: keep your old behavior for non-dynamic or non-token-mean
                        if self.config.use_dynamic_bsz:
                            # 这里如果你坚持要用旧逻辑，也可以；但严格来说 token-mean 下会有 bias
                            loss_scale_factor = n_rl / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        all_loss = pg_loss + self.config.sft_loss_coef * sft_loss
                        if entropy_coeff != 0 and calculate_rl_loss:
                            all_loss = all_loss - entropy_coeff * entropy_loss
                        loss = all_loss * loss_scale_factor
                    loss.backward()

                    if calculate_sft_loss:
                        micro_batch_metrics["actor/sft_loss"] = sft_loss.detach().item() if sft_loss is not None else 0
                        micro_batch_metrics["actor/sft_coef"] = self.config.sft_loss_coef

                    if calculate_rl_loss:
                        if self.config.use_dynamic_bsz and loss_agg_mode == "token-mean":
                            micro_batch_metrics["actor/pg_loss_weighted"] = (w_rl * pg_loss).detach().item()
                            micro_batch_metrics["actor/pg_loss_raw"] = pg_loss.detach().item()
                            micro_batch_metrics["actor/w_rl"] = float(w_rl) if not isinstance(w_rl, float) else w_rl
                        else:
                            micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor


                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics["actor/grad_norm"] = grad_norm.detach().item()
                mini_batch_metrics["actor/total_rl_tokens"] = total_rl_tokens.item()
                mini_batch_metrics["actor/total_rl_tokens_ratio"] = total_rl_tokens.item() / (total_rl_tokens.item() + total_sft_tokens.item())
                mini_batch_metrics["actor/total_sft_tokens"] = total_sft_tokens.item()
                mini_batch_metrics["actor/total_sft_tokens_ratio"] = total_sft_tokens.item() / (total_rl_tokens.item() + total_sft_tokens.item())
                # print(f"mini batch {batch_idx} grad norm: ", grad_norm)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics