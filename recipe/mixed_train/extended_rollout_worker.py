import datetime
import logging
import os

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_name, get_nccl_backend
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import RolloutConfig, HFModelConfig
from verl.workers.rollout.rollout_worker import RolloutWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class ExtendedRolloutWorker(RolloutWorker):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig) -> None:
        super(RolloutWorker, self).__init__()
        import torch.distributed
        from torch.distributed.device_mesh import init_device_mesh

        self.device_name = get_device_name()

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{self.device_name}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.config = config
        self.model_config = model_config

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            self.device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        rollout_name = self.config.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # build rollout engine here
        if self.config.name == "vllm":
            from recipe.mixed_train.extended_vllm_rollout import ExtendedVLLMRollout

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self.model_config.lora_rank}}
                if self.model_config.lora_rank > 0
                else {}
            )
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = ExtendedVLLMRollout if self.config.mode == "sync" else vLLMAsyncRollout
            self.rollout = vllm_rollout_cls(
                model_path=self.model_config.local_path,
                config=self.config,
                tokenizer=self.model_config.tokenizer,
                model_hf_config=self.model_config.hf_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=self.model_config.trust_remote_code,
                **lora_kwargs,
            )
        elif self.config.name == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            self.rollout = SGLangRollout(
                actor_module=self.model_config.local_path,
                config=self.config,
                processing_class=self.model_config.get_processor(),
                model_hf_config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        else:
            raise ValueError(f"Unknown rollout name: {self.config.name}")


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_se_blocks(self, prompts: DataProto):
        """Given a batch of prompts, return a batch of responses. Internally, it can use"""
        meta_info = {
            "eos_token_id": self.model_config.generation_config.eos_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.eos_token_id,
            "pad_token_id": self.model_config.generation_config.pad_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.rollout.generate_se_blocks(prompts=prompts)
        return output