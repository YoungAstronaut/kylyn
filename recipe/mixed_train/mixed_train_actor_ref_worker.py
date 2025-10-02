import types

from tensordict import TensorDict

from verl import DataProto
from verl.utils import omega_conf_to_dataclass
from verl.utils.profiler.performance import reduce_timing, GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.config import RolloutConfig, HFModelConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import logging
import os
import torch
import numpy as np

from omegaconf import OmegaConf, open_dict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import (
    get_device_name, get_device_id, get_torch_device,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import log_gpu_memory_usage, DistProfiler, simple_timer
from verl.workers.rollout.rollout_worker import RolloutWorker
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

@GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
@torch.no_grad()
def generate_sft_blocks(self, prompts: DataProto, **kwargs) -> DataProto:
    """Generate sequences for a batch of prompts.

    Args:
        batch (DataProto): Input batch.

    Returns:
        DataProto: Output batch.
        - prompts: [bsz, prompt_length], prompt token ids from dataset.
        - responses: [bsz, response_length], output token ids include response tokens
          from LLM generation and observation tokens from tool_calls.
        - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
        - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
          and response tokens.
        - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
        - position_ids: [bsz, prompt_length + response_length], incremental position ids.

        For multi-turn conversations:
        responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
        response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
    """
    idx = prompts.batch["input_ids"]  # (bs, prompt_length)
    # left-padded attention_mask
    attention_mask = prompts.batch["attention_mask"]

    # used to construct attention_mask
    eos_token_id = prompts.meta_info["eos_token_id"]

    batch_size = idx.size(0)

    non_tensor_batch = prompts.non_tensor_batch
    if "raw_prompt_ids" not in non_tensor_batch:
        non_tensor_batch["raw_prompt_ids"] = np.array(
            [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
        )

    if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
        raise RuntimeError("vllm sharding manager is not work properly.")

    vllm_inputs = [
        {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
    ]

    # ensure the type of `prompt_token_ids` passed to vllm is list[int]
    # https://github.com/volcengine/verl/pull/772
    for input_data in vllm_inputs:
        if isinstance(input_data["prompt_token_ids"], np.ndarray):
            input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
        elif not isinstance(input_data["prompt_token_ids"], list):
            raise TypeError(
                f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
            )

    max_tokens = prompts.meta_info.get("max_tokens", -1)
    if max_tokens > 0:
        kwargs = {"max_tokens": max_tokens}

    # users can customize different sampling_params at different run
    with self.update_sampling_params(**kwargs):
        outputs = self.inference_engine.generate(
            prompts=vllm_inputs,  # because we have already convert it to prompt token id
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        rollout_log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)
        # print(f' response: {response}')

        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=max_tokens).to(
            idx.device
        )
        # print(f' response {response}, shape: {response.shape}')


    # TODO(sgm): fix position_ids on right_pad
    # prompt: left pad + response: right pad
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
    response_attention_mask = get_response_mask(
        response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
    )

    # all the tp ranks should contain the same data here. data in all ranks are valid
    batch = TensorDict(
        {
            "responses": response,
            "response_attention_mask": response_attention_mask,
        },
        batch_size=batch_size,
    )
    if self.config.calculate_log_probs:
        # we will recompute old log prob with actor
        batch["rollout_log_probs"] = rollout_log_probs

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class MixedTrainActorRefWorker(ActorRolloutRefWorker):

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        # build rollout worker inside hybrid engine
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        rollout_worker = RolloutWorker(config=rollout_config, model_config=model_config)
        rollout_worker.generate_sft_blocks = types.MethodType(generate_sft_blocks, rollout_worker)
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        if rollout_name == "vllm":
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout_worker.rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout_worker.rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout_worker, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        from recipe.mixed_train.mixed_actor import MixedTrainParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                )
            )

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = MixedTrainParallelPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = MixedTrainParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="generate_sft_block")
    def generate_sft_blocks(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
            "max_tokens": self.config.rollout.self_explain.max_tokens,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}

        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            vllm_inputs = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sft_blocks(vllm_inputs)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output