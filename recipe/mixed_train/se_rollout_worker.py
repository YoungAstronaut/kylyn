import datetime
import logging
import os

import numpy as np
import torch
import torch.distributed
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM
from vllm import SamplingParams

import verl.utils.torch_functional as verl_F
from recipe.mixed_train.extended_rollout_worker import ExtendedRolloutWorker
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    init_fn,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.workers.config import RolloutConfig, HFModelConfig
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

class SERolloutWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "se_worker", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("se_worker", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        self.rollout = None
        self.se_module = None
        self.model_config = None
        self.pad_token_id = None

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
            self._register_dispatch_collect_info("se_rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "se_rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        print("*** rollout config: ", self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        # build rollout worker inside hybrid engine
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        rollout_worker = ExtendedRolloutWorker(config=rollout_config, model_config=model_config)
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        if rollout_name == "vllm":
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.se_module,
                inference_engine=rollout_worker.rollout.inference_engine,
                model_config=self.model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=False,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.se_module,
                inference_engine=rollout_worker.rollout._engine,
                model_config=self.model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=False,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout_worker, rollout_sharding_manager

    def _build_model(self):
        # fsdp_config = OmegaConf.create()
        fsdp_config = self.config.model.get("fsdp", {})
        override_model_config = OmegaConf.create()
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        local_path = copy_to_local(self.config.model.path, use_shm=False)
        self.se_tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        self.pad_token_id = self.se_tokenizer.pad_token_id
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
            "bos_token_id": self.se_tokenizer.bos_token_id,
            "eos_token_id": self.se_tokenizer.eos_token_id,
            "pad_token_id": self.se_tokenizer.pad_token_id,
        }
        override_model_config.update(override_config_kwargs)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        se_module = AutoModelForCausalLM.from_pretrained(
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
            model=se_module,
            use_remove_padding=False,
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
            use_fused_kernels=False,
            fused_kernels_backend=fused_kernels_backend,
        )
        se_module.to(torch_dtype)
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=se_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        cpu_offload = None
        se_module_fsdp = FSDP(
            se_module,
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
        return se_module_fsdp._fsdp_wrapped_module, model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.se_module, self.model_config = self._build_model()
        self.rollout = self._build_rollout()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="se"))
    @DistProfiler.annotate(color="brown")
    def generate_se_blocks(self, prompts: DataProto):
        idx = prompts.batch["input_ids"]
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

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # users can customize different sampling_params at different run
        with self.update_sampling_params():
            new_sampling_params = {'n': self.sampling_params.n,
                                   'logprobs': self.sampling_params.logprobs,
                                   'max_tokens': 200,
                                   'detokenize': False,
                                   'temperature': self.sampling_params.temperature,
                                   'top_p': self.sampling_params.top_p,
                                   'top_k': self.sampling_params.top_k,
                                   'ignore_eos': self.sampling_params.ignore_eos}
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already converted it to prompt token id
                sampling_params=SamplingParams(**new_sampling_params),
                use_tqdm=False,
            )

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)

            response = verl_F.pad_2d_list_to_length(response, self.pad_token_id, max_length=200).to(
                idx.device
            )

        response_attention_mask = verl_F.get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )

        batch = TensorDict(
            {
                "responses": response,
                "attention_mask": response_attention_mask,
                "input_ids": idx,
                "input_attention_mask": attention_mask,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)