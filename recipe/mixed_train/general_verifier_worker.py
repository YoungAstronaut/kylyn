# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The main entry point to run the PPO algorithm
"""

import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from typing import Any

import ipdb
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import verl.utils.torch_functional as verl_F
from tests.special_e2e.envs.digit_completion.task import compute_position_id_with_mask
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.profiler import DistProfiler
from verl.workers.fsdp_workers import RewardModelWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

class GeneralVerifierWorker(RewardModelWorker):
    def __init__(self, config):
        super().__init__(config)
        self.rollout = None
        self.rollout_sharding_manager = None
        self.reward_module = None
        self.generation_config = None
        self.rm_processor = None
        self.rm_tokenizer = None
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer.
        """
        fsdp_config = OmegaConf.create()
        override_model_config = OmegaConf.create()
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        local_path = copy_to_local(self.config.model.path, use_shm=False)
        self.rm_tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        self.rm_processor = hf_processor(
            self.config.model.path,
            trust_remote_code=trust_remote_code
        )
        torch_dtype = torch.bfloat16
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2"
        )
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        override_config_kwargs = {
            "bos_token_id": self.rm_tokenizer.bos_token_id,
            "eos_token_id": self.rm_tokenizer.eos_token_id,
            "pad_token_id": self.rm_tokenizer.pad_token_id,
        }
        override_model_config.update(override_config_kwargs)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        reward_module = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            trust_remote_code=trust_remote_code,
        )
        fused_kernel_options = self.config.model.get("fused_kernel_options", None)
        fused_kernels_backend = (
            fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
        )
        apply_monkey_patch(
            model=reward_module,
            use_remove_padding=False,
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
            use_fused_kernels=False,
            fused_kernels_backend=fused_kernels_backend,
        )
        reward_module.to(torch_dtype)
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        cpu_offload = None
        reward_module_fsdp = FSDP(
            reward_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=False,
            forward_prefetch=False,
        )
        self.reward_module = reward_module_fsdp._fsdp_wrapped_module

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        from verl.workers.rollout.vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        self.rollout = vLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.rm_tokenizer,
            model_hf_config=model_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
        )

        full_params = torch.distributed.get_world_size() == 1
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=reward_module_fsdp,
            inference_engine=self.rollout.inference_engine,
            model_config=model_config,
            rollout_config=self.config.rollout,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.get("layered_summon", False),
        )

        torch.cuda.empty_cache()

    def _formulate_responses_for_reward(self, data: DataProto):
        batch_size = data.batch['input_ids'].shape[0]

        reward_model_prompts = []
        for i in range(batch_size):
            raw_prompt = self.rm_tokenizer.decode(data.batch['prompts'][i], skip_special_tokens=True)
            question = re.findall(r'\nuser(.*?)\nassistant', raw_prompt, re.DOTALL)[-1].strip()
            response = self.rm_tokenizer.decode(data.batch['responses'][i], skip_special_tokens=True)
            try:
                student_answer = re.findall(r'\n<answer>\n(.*?)\n</answer>', response, re.DOTALL)[-1].strip()
            except IndexError:
                student_answer = "Not Found"
            ground_truth = data.non_tensor_batch['reward_model'][i]['ground_truth']
            prompt = (
                f"User: ### Question: {question}\n\n"
                f"### Ground Truth Answer: {ground_truth}\n\n"
                f"### Student Answer: {student_answer}\n\n"
                "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
                "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
                "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
            )
            reward_model_prompts.append(prompt)
        return reward_model_prompts

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        messages = self._formulate_responses_for_reward(data)

        model_inputs = self.rm_tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            max_length=1024,
            truncation=True,
        )
        inputs_id, attention_mask = verl_F.postprocess_data(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_length=1024,
            pad_token_id=self.rm_tokenizer.pad_token_id,
            left_pad=True,
            truncation='left'
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        reward_model_inputs = DataProto.from_single_dict(
            {
                "input_ids": inputs_id,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        reward_model_inputs = reward_model_inputs.to(get_device_id())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.rm_tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.rm_tokenizer.pad_token_id,
        }
        reward_model_inputs.meta_info.update(meta_info)

        with self.rollout_sharding_manager:
            # ipdb.set_trace()
            reward_model_inputs = self.rollout_sharding_manager.preprocess_data(reward_model_inputs)
            reward_model_outputs = self.rollout.generate_sequences(prompts=reward_model_inputs)
            outputs = self.rollout_sharding_manager.postprocess_data(reward_model_outputs)
        batch_size = outputs.batch['input_ids'].shape[0]
        responses_texts = []
        for i in range(batch_size):
            responses_texts.append(self.rm_tokenizer.decode(outputs.batch['responses'][i], skip_special_tokens=True))
        outputs.non_tensor_batch.update({"reward_model_responses": responses_texts})
        outputs = outputs.to('cpu')

        # clear kv cache
        get_torch_device().empty_cache()

        return outputs