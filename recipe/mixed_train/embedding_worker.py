import datetime
import inspect
import logging
import os

import ray
import torch
import torch.distributed

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
    get_nccl_backend, get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.utils.ray_utils import get_event_loop
from verl.workers.fsdp_workers import create_device_mesh

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

class EmbeddingWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.model.fsdp_config.fsdp_size)

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        self.max_blocks_num = self.config.model.get("max_blocks_num", 8)

        self.sleep_level = VLLM_SLEEP_LEVEL

        self.gen_random_states = None

    def _build_model(self, config):
        print(f"[Worker PID {os.getpid()}] 进程启动后第一条日志")
        print(f"[Worker PID {os.getpid()}] PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        print(f"[Worker PID {os.getpid()}] torch.cuda.is_initialized(): {torch.cuda.is_initialized()}")


        # the following line is necessary
        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.embedding_model.path, use_shm=use_shm)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))
        tensor_parallel_size = self.config.get("tensor_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )

        from vllm import LLM
        print(f"[{os.getpid()}] Starting LLM initialization...")
        self.inference_engine = LLM(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            task='embed',
        )
        print(f"[{os.getpid()}] LLM initialization completed!")

    async def fall_into_sleep(self):
        if self.config.free_cache_engine:
            log_gpu_memory_usage("Before embedding model offload", logger=logger)
            await self.release()
            log_gpu_memory_usage("After embedding model offload", logger=logger)

        aggressive_empty_cache(force_sync=True)

    async def wake_up(self):
        aggressive_empty_cache(force_sync=True)
        if self.config.free_cache_engine:
            log_gpu_memory_usage("Before embedding model wake up", logger=logger)
            await self.resume(tags=["weights", "kv_cache"])
            log_gpu_memory_usage("After embedding model wake up", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self._build_model(config=self.config)
        loop = get_event_loop()
        loop.run_until_complete(self.fall_into_sleep())

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    @DistProfiler.annotate(color="brown")
    def calculate_similarity(self, data: list[list[dict]]):
        print("calculate_similarity EmbeddingWorker PID", os.getpid(), "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"))
        steps_data_list = []
        for data_item in data:
            steps_data_list.extend(data_item)
        print('length of steps_data_list: ', len(steps_data_list))

        steps_data_length = []
        all_input_texts = []
        for steps_data in steps_data_list:
            if steps_data["pair_idx"] == -1:
                steps_data_length.append(0)
            else:
                steps_data_length.append(len(steps_data["texts"]))
                all_input_texts.extend(steps_data["texts"])
        assert len(steps_data_length) == len(steps_data_list)

        outputs = self.inference_engine.embed(all_input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])

        offset = 0
        scores_list = []
        for steps_data, steps_length in zip(steps_data_list, steps_data_length):
            if steps_length == 0:
                scores_list.append({"scores": []})
                continue
            m, k = steps_data["m"], steps_data["k"]
            pair_embeddings = embeddings[offset:offset + m + k]

            steps_embs = pair_embeddings[:m]
            answers_embs = pair_embeddings[m:]
            steps_embs = torch.nn.functional.normalize(steps_embs, p=2, dim=1)
            answers_embs = torch.nn.functional.normalize(answers_embs, p=2, dim=1)
            scores = steps_embs @ answers_embs.T # (m, k)
            scores = scores.max(dim=1) # (m,) TODO: 这里当k=1的时候相当于没效果，且是合理的，当k不为1的时候需要考虑这样处理的合理性

            scores_list.append({"scores": scores.values.tolist(), "extra_info": steps_data.copy()})
            offset += m + k
        assert offset == len(all_input_texts)
        return scores_list