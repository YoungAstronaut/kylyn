import datetime
import logging
import os

import torch
import torch.distributed

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
)
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler, DistProfilerExtension
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
            # torch.distributed.init_process_group(
            #     backend=get_nccl_backend(), init_method=os.environ.get("DIST_INIT_METHOD", None)
            # )
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

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        self.max_blocks_num = self.config.model.get("max_blocks_num", 8)

    def _build_model(self, config):
        # the following line is necessary
        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.embedding_model.path, use_shm=use_shm)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))
        tensor_parallel_size = self.config.get("tensor_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        import os

        os.environ["VLLM_USE_V1"] = "0"

        from vllm import LLM
        self.inference_engine = LLM(
            model=local_path,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            # enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            # disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # enable_prefix_caching=False,
            # trust_remote_code=False,
            seed=config.get("seed", 0),
            task='embed',
        )
        print("EmbeddingWorker PID", os.getpid(), "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"))
        # self.inference_engine.sleep(2)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self._build_model(config=self.config)

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