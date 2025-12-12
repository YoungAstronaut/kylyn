from verl import DataProto
from verl.utils import omega_conf_to_dataclass
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import logging
import os

from omegaconf import DictConfig

from verl.single_controller.base.decorator import Dispatch, register, make_nd_compute_dataproto_dispatch_fn
from verl.utils.device import (
    get_device_name,
)
from verl.utils.fsdp_utils import (
    fsdp_version,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
)
from verl.utils.profiler import log_gpu_memory_usage, DistProfiler

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class MixedTrainActorRefWorker(ActorRolloutRefWorker):

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self.actor = None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()

        # 获取eos_token_id
        eos_token_id = data.meta_info.pop("eos_token_id", -1)
        assert eos_token_id >= 0, 'please specify eos_token_id'

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys, eos_prob = self.actor.compute_log_prob(
                    data=data, calculate_entropy=True, need_eos_prob=True, eos_token_id=eos_token_id) # 更改了调用参数
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys, "eos_prob": eos_prob},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        from recipe.mixed_train.mixed_actor import MixedTrainParallelPPOActor

        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = MixedTrainParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )