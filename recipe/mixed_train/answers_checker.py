import logging
import os

import numpy as np

import torch
import torch.distributed

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend, get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.workers.fsdp_workers import create_device_mesh

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def np_split_by_sizes(arr, sizes, axis=0):
    sizes = list(map(int, sizes))
    if any(s < 0 for s in sizes):
        raise ValueError("sizes 里不能有负数")
    if sum(sizes) != arr.shape[axis]:
        raise ValueError(f"sizes 之和必须等于被分割维度长度 {arr.shape[axis]}")
    idx = np.cumsum(sizes)[:-1]          # 关键：去掉最后一个总和
    return np.split(arr, idx, axis=axis)

class AnswersChecker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(), init_method=os.environ.get("DIST_INIT_METHOD", None)
            )
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        self.task = ('Given a query of a math question and the standard answer, '
                     'conclude whether the query is related to the standard answer.')

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
            enable_sleep_mode=False,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.max_model_len,
            load_format="dummy" if config.load_format.startswith("dummy") else config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=False,
            seed=config.get("seed", 0),
            task='embed',
        )
        print("AnswersChecker PID", os.getpid(), "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"))
        # self.inference_engine.sleep(2)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self._build_model(config=self.config)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def generate_answers_mask(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        # print(f' data: {data}')

        # self.inference_engine.wake_up()

        queries = data.non_tensor_batch["queries"].tolist()
        # print(f'type of queries: {type(queries)}')

        blocks_length = data.non_tensor_batch["blocks_length"].tolist()
        blocks_sum = sum(blocks_length)

        blocks = data.non_tensor_batch["parsed_blocks"].tolist()
        # print(f"type of blocks: {type(blocks)}")

        queries_all = []
        for item in queries:
            queries_all += item
        # print(f' queries all: {queries_all}')
        complete_answers = data.non_tensor_batch["raw_tgt_prompts"].tolist()
        input_texts = queries_all + complete_answers
        # print(f"input queries length: {len(input_texts)}")
        outputs = self.inference_engine.embed(input_texts, use_tqdm=False)
        embeddings = torch.tensor([output.outputs.embedding for output in outputs])
        documents = embeddings[blocks_sum:]

        filtered_sum = 0
        filtered_blocks = []
        # print(f'blocks length: {blocks_length}, blocks sum: {blocks_sum}')
        embeddings_splits = embeddings[:blocks_sum].split(blocks_length, dim=0)
        input_texts = np.array(queries_all)
        input_texts_splits = np_split_by_sizes(input_texts, blocks_length)
        assert len(embeddings_splits) == len(blocks), f'{len(embeddings_splits)} != {len(blocks)}'
        assert len(embeddings_splits) == len(blocks_length), f'{len(embeddings_splits)} != {len(blocks_length)}'
        assert len(input_texts_splits) == len(blocks_length), f'{len(embeddings_splits)} != {len(input_texts_splits)}'

        for idx, embeddings_split in enumerate(embeddings_splits):
            block_out = []
            if blocks_length[idx] == 0:
                filtered_blocks.append([])
                continue
            scores = embeddings_split @ documents[idx:idx+1].T
            # print('shape of scores: ', scores.shape)
            filtered_scores_index = (scores > self.config.similarity_threshold).nonzero(as_tuple=True)[0].tolist()

            filtered_sum += len(filtered_scores_index)
            # print(f'filtered scores index: {filtered_scores_index}, length: {len(filtered_scores_index)}')

            for filter_idx in filtered_scores_index:
                start = blocks[idx][filter_idx].start
                end = blocks[idx][filter_idx].end
                print(input_texts_splits[idx][filter_idx], f'score: {scores[filter_idx, 0]:.4f}')
                block_out.append([start, end])
            # print(block_out)
            filtered_blocks.append(block_out)

        assert len(filtered_blocks) == len(blocks_length), f'{len(filtered_blocks)} != {len(blocks_length)}'

        filtered_ratio = filtered_sum / blocks_sum
        print(f'filtered ratio: {filtered_ratio}')

        # data.non_tensor_batch["filtered_blocks"] = np.array(filtered_blocks)
        data.non_tensor_batch["filtered_blocks"] = np.array([np.asarray(b, np.int32).reshape(-1, 2) for b in filtered_blocks],
                   dtype=object)

        output = data.to("cpu")

        get_torch_device().empty_cache()
        return output
