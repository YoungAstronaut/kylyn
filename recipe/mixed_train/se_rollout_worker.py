import logging
import os

from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
)
from verl.utils.profiler import DistProfiler
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

class SERolloutWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, 'rollout', **kwargs)
        print('finished initializing se rollout worker')
        print(f'is actor: {self._is_actor}, is rollout: {self._is_rollout}, is ref: {self._is_ref}')

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="brown", role="se_rollout_generate")
    def generate_se_blocks(self, prompts: DataProto):
        output = self.generate_sequences(prompts)
        batch = TensorDict(
            {
                "responses": output.batch["responses"],
                "attention_mask": output.batch["response_mask"],
                "input_ids": prompts.batch["input_ids"],
                "input_attention_mask": prompts.batch["attention_mask"],
            }
        )
        return DataProto(batch=batch, non_tensor_batch=output.non_tensor_batch)