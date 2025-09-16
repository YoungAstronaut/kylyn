import logging
import os
import numpy as np

import torch
import torch.distributed

from recipe.mixed_train.semantic_blocks import build_high_entropy_blocks_tensor
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
)
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.workers.fsdp_workers import create_device_mesh
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

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
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

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
        import os, sys

        os.environ["VLLM_USE_V1"] = "0"

        from vllm import LLM
        self.inference_engine = LLM(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
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
            enable_prefix_caching=True,
            trust_remote_code=False,
            seed=config.get("seed", 0),
            task='embed',
        )
        print("AnswersChecker PID", os.getpid(), "VLLM_USE_V1=", os.environ.get("VLLM_USE_V1"))
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

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

        responses = data.batch['responses']
        advantages = data.batch['advantages']
        entropys = data.batch['entropys']
        response_mask = data.batch['response_mask']
        advantages_sum = advantages.sum(dim=-1)
        positive_nums = (advantages_sum >= 0).sum().item()
        # print(f' number of negative rewards:  {positive_nums}')
        process_mask = (advantages < 0).int() * response_mask
        process_mask_sum = (process_mask.sum(dim=-1) == 0).int()
        # print(f' process_mask_sum:  {process_mask_sum.sum().item()}')
        # print(f' correct ratio: {positive_nums / advantages_sum.shape[0]}')

        flat_responses = responses.view(-1).tolist()
        flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_responses)
        flat_tokens = [token.replace("Ġ", " ").replace("Ċ", "\n") if token is not None else "" for token in flat_tokens]
        tokens = [flat_tokens[i*responses.size(1):(i+1)*response_mask.size(1)] for i in range(responses.size(0))]
        blocks = build_high_entropy_blocks_tensor(tokens, entropys, process_mask*response_mask, seed_method='mean_std',
                                                  max_block_len=16, min_block_len=3, stop_on_sentence_boundary=True)
        queries = []
        blocks_length = []
        for block_list in blocks:
            blocks_length.append(len(block_list))
            for block in block_list:
                queries.append(get_detailed_instruct(task_description=self.task, query=block.text))
        blocks_sum = sum(blocks_length)
        complete_answers = data.non_tensor_batch["raw_tgt_prompts"].tolist()
        input_texts = queries + complete_answers
        print(f"input queries length: {len(input_texts)}")
        outputs = self.inference_engine.embed(input_texts, use_tqdm=False)
        embeddings = torch.tensor([output.outputs.embedding for output in outputs])
        documents = embeddings[blocks_sum:]

        filtered_sum = 0
        filtered_blocks = []
        # print(f'blocks length: {blocks_length}, blocks sum: {blocks_sum}')
        embeddings_splits = embeddings[:blocks_sum].split(blocks_length, dim=0)
        input_texts = np.array(queries)
        input_texts_splits = np_split_by_sizes(input_texts, blocks_length)
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
                process_mask[idx, ...] = 0
                start = blocks[idx][filter_idx].start
                end = blocks[idx][filter_idx].end
                process_mask[idx, start:end] = 1
                # print(input_texts_splits[idx][filter_idx], f'score: {scores[filter_idx, 0]:.4f}')
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
        return output
