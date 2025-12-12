"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import json
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
from typing import Optional, Any, Tuple

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from openai import OpenAI
from tensordict import TensorDict
from torch.utils.data import Sampler
import torch.nn.functional as F
from tqdm import tqdm

from recipe.mixed_train.embed_utils import balance_embeddings_batch, \
    TASK_PREFIX, find_first_descent_point, argmin
from recipe.mixed_train.semantic_blocks import build_high_entropy_blocks_tensor, Block, split_into_blocks, \
    text_to_pieces
from recipe.mixed_train.step_localization import localize_first_error_chat
from verl import DataProto
from verl.protocol import unpad_dataproto, pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls, RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask, ResourcePoolManager, WorkerType,
)
from verl.trainer.ppo.utils import Role
from verl.utils import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.torch_functional import pad_sequence_to_length


def pad_sequence_to_length_with_trunc(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors[:, :max_seq_len]
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)

def get_embeddings_via_server(texts, base_url="http://127.0.0.1:8005/v1",
                              model_name="qwen3-embed-4b", api_key="secret-embed-key"):
    """
    texts: list[str]
    返回: torch.FloatTensor [N, D]
    """
    client = OpenAI(base_url=base_url, api_key=api_key)  # OpenAI 兼容
    resp = client.embeddings.create(model=model_name, input=texts)
    embs = [item.embedding for item in resp.data]
    return torch.tensor(embs, dtype=torch.float32)

def localize_error_by_llm(
    blocks: list[list[Block]],
    complete_answers: list[str],
    questions: list[str],
    acc_labels: list[bool],
    max_workers: Optional[int] = None,
    verbose: bool = True,
):
    """
    使用 LLM 对每个样本的思维链分块进行复核，定位第一处错误的 block。

    Args:
        blocks: 每个样本的一组 Block 列表，形如 list[样本][Block]。
        questions: 每个样本的问题文本。
        complete_answers: 每个样本的完整参考答案（一般是标准答案或高质量答案）。
        acc_labels: 每个样本的原始判定是否正确（True 表示原本就正确）。
        max_workers: 线程池最大并发数；为 None 或 <=0 时自动估计。
        verbose: 是否打印每个 worker 的耗时与 verdict。

    Returns:
        error_blocks:
            长度为 n 的列表；
            - 对于经 LLM 复核后仍被判为错误的样本：为 (start, end, idx)；
            - 对于被 LLM 判为正确的样本：为 None；
            - 对于原本就正确且未请求 LLM 的样本：也为 None。
        re_verified_true:
            长度为 n 的 bool 列表；
            - True  表示经 LLM 复核认为“整体正确”；
            - False 表示 LLM 认为存在错误或请求失败。
        llm_results:
            长度为 n 的列表，对应每个样本的原始 LLM verdict（或错误信息）；
            - 未调用 LLM 的样本为 None。
    """

    # ------------------ 基本输入检查 ------------------ #
    n = len(blocks)
    if not (len(questions) == len(complete_answers) == len(acc_labels) == n):
        raise ValueError(
            f"Inconsistent input lengths: "
            f"blocks={n}, questions={len(questions)}, "
            f"complete_answers={len(complete_answers)}, acc_labels={len(acc_labels)}"
        )

    # ——打印样例信息（安全防护：判空）——
    if len(questions) > 0:
        print(f"sample  question: {questions[0]}")
    if len(complete_answers) > 0:
        print(f"sample complete answer: {complete_answers[0]}")
    print(f"length of blocks: {len(blocks)}")
    print(f"num of correct answers: {acc_labels.count(True)}")

    # ------------------ 环境变量与 LLM client ------------------ #
    api_key = os.getenv("OPENAI_API_KEY", "REPLACE_ME")
    if api_key == "REPLACE_ME":
        print("⚠️ 未检测到环境变量 OPENAI_API_KEY，请设置后再运行。")

    base_url = os.getenv("BASE_URL", "NOT_SPECIFIED")
    if base_url == "NOT_SPECIFIED":
        print("⚠️ 未检测到环境变量 BASE_URL，请设置后再运行。")

    # 如果希望缺配置时直接终止，可以换成 raise
    if api_key == "REPLACE_ME" or base_url == "NOT_SPECIFIED":
        raise RuntimeError("OPENAI_API_KEY 或 BASE_URL 未设置。")

    # 默认复用一个 client；如果确认该客户端非线程安全，可以挪到 worker 内部重建
    llm_client = OpenAI(base_url=base_url, api_key=api_key)

    # ------------------ 输出容器（与输入样本一一对应） ------------------ #
    error_blocks: list[Optional[Tuple[int, int, int]]] = [None] * n
    re_verified_true: list[bool] = [False] * n
    # 用 None 表示“尚未填充”，避免 [{}] * n 的共享引用坑
    llm_results: list[Optional[Any]] = [None] * n

    # ------------------ 选择需要调用 LLM 的样本 ------------------ #
    # 只对原本判为错误的样本调用 LLM，节省请求配额
    indices_to_run = [i for i, ok in enumerate(acc_labels) if not ok]
    print(f"indices_to_run: {indices_to_run}")

    # 如果所有样本原本都正确，则无需调用 LLM，直接返回默认结果
    if not indices_to_run:
        print("All samples already correct, skip LLM calls.")
        print(f"num of re correct answers: {re_verified_true.count(True)}")
        return error_blocks, re_verified_true, llm_results

    # ------------------ 单样本处理函数（在工作线程里执行 I/O） ------------------ #
    def _process_one(i: int):
        """
        对第 i 个样本调用 localize_first_error_chat，返回：
            (i, err_tuple, re_true, verdict, elapsed)
        其中：
            err_tuple: (start, end, idx) 或 None
            re_true: 是否被 LLM 复核为“整体正确”
            verdict: LLM 的原始返回或错误信息
            elapsed: 本样本耗时（秒）
        """
        start_time = time.time()
        try:
            # 如需线程安全 client，可在此处重新构建：
            # client = OpenAI(base_url=base_url, api_key=api_key)
            client = llm_client

            # 提取当前样本的思维链文本序列
            steps = [blk.text for blk in blocks[i]]

            # 调用 LLM 进行错误定位
            verdict_single = localize_first_error_chat(
                questions[i],
                steps,
                reference_answer=complete_answers[i],
                client=client,
            )
            elapsed_single = time.time() - start_time

            # verdict_single 期望是一个 dict，包含 "k" 表示第一处错误的 1-based index
            if isinstance(verdict_single, dict) and verdict_single.get("k"):
                try:
                    # 尝试解析 k 并转成 0-based index
                    idx = int(verdict_single["k"]) - 1
                except (TypeError, ValueError):
                    # k 字段格式异常，视为“无法定位错误”，按整体正确处理
                    return i, None, True, verdict_single, elapsed_single

                # 越界保护：如果 k 不在合法范围内，也视为整体正确，避免 IndexError
                if 0 <= idx < len(blocks[i]):
                    blk = blocks[i][idx]
                    err_tuple = (blk.start, blk.end, idx)
                    return i, err_tuple, False, verdict_single, elapsed_single
                else:
                    # k 超出 steps 数量，认为定位无效，按整体正确处理
                    return i, None, True, verdict_single, elapsed_single
            else:
                # 没有 k 或 verdict 不是 dict，则视为“未发现第一处错误”→ 整体正确
                return i, None, True, verdict_single, elapsed_single

        except Exception as e:
            # 任何异常统一收敛成一个错误 verdict，以免线程异常直接冒到主线程
            elapsed_single = time.time() - start_time
            err_msg = {"error": str(e), "worker": i}
            return i, None, False, err_msg, elapsed_single

    # ------------------ 并发执行 ------------------ #
    if not max_workers or max_workers <= 0:
        import os as _os

        cpu_cnt = _os.cpu_count() or 4
        # I/O 密集任务：可以适当放大线程数，但不宜过大
        max_workers = min(128, cpu_cnt) * 5
        print(f"cpu_count: {cpu_cnt}")
    print(f"max_workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # 提交任务，获得 Future 列表
        futures = [ex.submit(_process_one, i) for i in indices_to_run]

        # 主线程按完成顺序收集结果（写回时按样本索引位置）
        for fut in as_completed(futures):
            i, err_tuple, re_true, verdict, elapsed = fut.result()

            # 如果 verdict 是 dict，顺便标记该样本 index 方便排查
            if isinstance(verdict, dict):
                verdict.setdefault("worker", i)

            error_blocks[i] = err_tuple
            re_verified_true[i] = re_true
            llm_results[i] = verdict

            if verbose:
                print(f"[worker-{i}] costs time: {elapsed:.3f}s")
                if verdict is not None:
                    print(f"[worker-{i}] verdict: {verdict}")

    # 对于原本就正确的样本：error_blocks 仍为 None，re_verified_true 为 False（按原逻辑保留）
    print(f"num of re correct answers: {re_verified_true.count(True)}")

    return error_blocks, re_verified_true, llm_results

def construct_explain_prompt(question: str, standard_answer: str, answer_prefix: str):
    chat = [
        {
            "content": f"Your task is to understand a given standard problem solving process of a given question, "
                       f"then finish an incomplete reasoning process. The question is :\n{question}\nThe standard "
                       f"solving process is as followings: \n\"\n{standard_answer}\n\"\n",
            "role": "system"
        },
        {
            "content": f"**Finish the following incomplete answer**: \n{answer_prefix}",
            "role": "user"
        }
    ]
    # custom_tmpl = """{%- for m in messages -%}
    # {{ m['content'] }}
    #
    # {%- endfor -%}"""
    #
    # result = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False,
    #                                             chat_template=custom_tmpl)
    # print(chat)
    result = chat[0]["content"]+" User: "+chat[1]["content"]
    # k = result.replace('\n', '&&')
    # print(f' explain prompt: {k}')
    return result


class RayMixedTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, config, tokenizer, role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup, processor=None, reward_fn=None,
                 val_reward_fn=None, train_dataset: Optional[Dataset] = None, val_dataset: Optional[Dataset] = None,
                 collate_fn=None, train_sampler: Optional[Sampler] = None, device_name=None):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor,
                         reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)
        self.actor_rollout_wg = None
        self.rm_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None
        self.embedding_wg = None
        self.se_rollout_wg = None

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # 新添加的内容
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.EmbeddingWorker)
        emb_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.EmbeddingWorker],
                                       config=self.config.embedding_worker)
        self.resource_pool_to_cls[resource_pool][str(Role.EmbeddingWorker)] = emb_cls

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.SEWorker)
        se_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.SEWorker], config=self.config.se_rollout_worker)
        self.resource_pool_to_cls[resource_pool][str(Role.SEWorker)] = se_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name
        wg_kwargs["worker_env"] = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            # "VLLM_USE_RAY_SPMD_WORKER": "1",
        }

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            assert not torch.cuda.is_initialized(), "CUDA was initialized too early!"
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.embedding_wg = all_wg[str(Role.EmbeddingWorker)]
        self.embedding_wg.init_model()

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        self.se_rollout_wg = all_wg[str(Role.SEWorker)]
        self.se_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def localize_error_by_emb(self, blocks: list[list[Block]], complete_answers: list[str]) -> list[
        list[Any] | list[tuple[int, int]]]:
        # 获取steps对应的texts
        steps_list: list = []
        ref_steps_list = []
        for block_list, answer in zip(blocks, complete_answers):
            steps_list.append([])
            for block in block_list:
                steps_list[-1].append(block.text)
            if len(block_list) > 0:
                ref_steps_list.append([answer])
            else:
                ref_steps_list.append([])
        assert len(steps_list) == len(ref_steps_list)

        # 将steps改成前缀逐step增加
        for i in range(len(steps_list)):
            steps = steps_list[i]
            steps_num = len(steps)
            prefix = [''.join(steps[:i+1]) for i in range(steps_num)]
            steps_list[i] = prefix
        print(len(steps_list))

        aux_steps_list: list = [None] * len(steps_list) # 辅助列表
        err_idx_list: list = [-1] * len(steps_list)
        for i in range(len(steps_list)):
            if len(steps_list[i]) > 0:
                aux_steps_list[i] = steps_list[i].copy()
                err_idx_list[i] = 0

        finished_flag = False
        max_blocks_num = self.config.trainer.max_blocks_num
        while not finished_flag:
            print('none count: ', aux_steps_list.count(None))
            tmp_steps_list = []
            for i in range(len(aux_steps_list)):
                if aux_steps_list[i] is not None:
                    if len(aux_steps_list[i]) > max_blocks_num:
                        tmp_steps_list.append(aux_steps_list[i][:max_blocks_num])
                    else:
                        tmp_steps_list.append(aux_steps_list[i])
                else: # 表明这个步骤列表已经定位到了错误位置或者本来就是正确度
                    tmp_steps_list.append([])

            balanced_batch = balance_embeddings_batch(tmp_steps_list, ref_steps_list, TASK_PREFIX, tokenizer=self.tokenizer)

            print('balanced batch num: ', len(balanced_batch))
            # for item in balanced_batch:
            #     print(item)
            embedding_group_num = self.embedding_wg.world_size
            # print('group num: ', embedding_group_num)

            if self.config.trainer.get("split_blocks", False):
                raise NotImplementedError
                # TODO: 目前的机制word_group里的句子数量不均等的时候容易卡死因此分块加速暂时不支持
                # split_size: int = len(balanced_batch) // embedding_group_num + 1
                # print('split size: ', split_size)
                # batch_splits = [balanced_batch[i:i+split_size] for i in range(0, len(balanced_batch), split_size)]
                # splits_sentences_num = []
                # for split in batch_splits:
                #     k = []
                #     for t in split:
                #         k.extend(t)
                #     # print(f'length of this batch sentences: {len(k)}')
                #     tokens_sum_this_batch = 0 # 统计一下总tokens数
                #     sentences_num = 0 # 统计一下总句子数
                #     for item in k:
                #         texts = item["texts"]
                #         print('texts length: ', len(texts))
                #         sentences_num += len(texts)
                #         for text in texts:
                #             tokens_sum_this_batch += len(self.tokenizer.encode(text))
                #             print(len(self.tokenizer.encode(text)))
                #     print(f'sum of this batch sentences: {sentences_num}')
                #     splits_sentences_num.append(sentences_num)
                #     print(f'sum of this batch tokens: {tokens_sum_this_batch}')
                #
                # # 进行padding操作
                # max_sentences_num = max(splits_sentences_num)
                # for sentences_num, batch_split in zip(splits_sentences_num, batch_splits):
                #     if sentences_num < max_sentences_num:
                #         batch_split.append([{'pair_idx': -1, 'texts': ['pad text'] * (max_sentences_num - sentences_num)}])
            else:
                batch_splits = [balanced_batch] * embedding_group_num

            similarity_results = []
            outputs = self.embedding_wg.calculate_similarity(batch_splits)
            if self.config.trainer.get("split_blocks", False):
                for output in outputs:
                    similarity_results.extend(output)
            else:
                similarity_results = outputs[1]

            print('length of similarity results: ', len(similarity_results))
            print('empty similarity results: ', similarity_results.count({"scores": []}))

            for i, similarity_result in enumerate(similarity_results):
                scores = similarity_result["scores"]
                # print(scores)
                if len(scores) == 0:
                    continue
                else:
                    loc = find_first_descent_point(scores)
                    if loc == -1:
                        if len(aux_steps_list[i]) > max_blocks_num:
                            aux_steps_list[i] = aux_steps_list[i][max_blocks_num//2:]
                            err_idx_list[i] += max_blocks_num//2
                            # print('no descent point found and need next turn ', ' err: ', err_idx_list[i])
                        else:
                            aux_steps_list[i] = None
                            err_idx_list[i] += argmin(scores)
                            # print('no descent point found, ', argmin(scores), ' err: ', err_idx_list[i])
                    else:
                        aux_steps_list[i] = None
                        err_idx_list[i] += loc
                        # print('found descent point: ', loc, ' err: ', err_idx_list[i])

            # print('err idx list: ', err_idx_list)
            finished_flag = aux_steps_list.count(None) ==len(aux_steps_list)

        error_blocks = []
        assert len(err_idx_list) == len(blocks)
        for i, block_list in enumerate(blocks):
            if err_idx_list[i] == -1:
                error_blocks.append(None)
                continue
            error_blocks.append(block_list[err_idx_list[i]])
        return error_blocks

    def divide_answers_blocks(self, data: DataProto, reward_tensor=None, eos_probs=None):
        """
        从 data 中的回答序列里，根据 token 熵和 reward 选出需要处理的高熵 token block。
        只对“整体 reward < 0”的样本进行 block 提取。
        最终结果写回 data.non_tensor_batch['parsed_blocks']。
        """

        # responses: (batch_size, seq_len) 的 token id
        responses: torch.Tensor = data.batch['responses']
        batch_size, seq_len = responses.shape

        # -----------------------------
        # 1. 计算每个样本的“序列级 reward”，再广播到每个 token
        # -----------------------------
        if reward_tensor is None:
            # data.batch['token_level_scores']: 一般是 (B, T_token)，在 dim=-1 上求和得到每个样本一个标量
            seq_scores = data.batch['token_level_scores'].sum(dim=-1, keepdim=True)  # (B, 1)
        else:
            # reward_tensor 可能是 (B, T) 或 (B,)
            if reward_tensor.dim() == 1:
                # (B,) -> (B, 1)
                seq_scores = reward_tensor.unsqueeze(1)
            else:
                # (B, T_token) -> (B, 1)
                seq_scores = reward_tensor.sum(dim=-1, keepdim=True)

        # 把每个样本的序列级 reward 广播到 seq_len 长度，得到 (B, seq_len)
        seq_scores = seq_scores.expand(-1, seq_len)

        # 2. 构造各种 mask
        # 有效 token（排除 PAD 和 EOS），形状 (B, seq_len)
        response_mask = (
                (responses != self.tokenizer.pad_token_id) &
                (responses != self.tokenizer.eos_token_id)
        ).to(dtype=torch.int32)

        # 只处理“序列总 reward < 0”的样本中的有效 token
        negative_seq_mask = (seq_scores < 0).to(dtype=torch.int32)  # (B, seq_len)
        process_mask = negative_seq_mask * response_mask  # (B, seq_len)

        # 3. 取出熵，并做一下形状校验
        entropys: torch.Tensor = data.batch['entropys']  # 一般是 (B, seq_len)
        # 如果这里 shape 对不上，说明上游就有问题，直接 assert 出来方便排查
        assert entropys.shape[:2] == responses.shape, \
            f"entropys.shape {entropys.shape} must match responses.shape {responses.shape}"

        # 4. 还原成按 batch 切分的二维 list: list[list[str]]，每一行长度为 seq_len
        tokens = []
        for i in range(batch_size):
            sentence = self.tokenizer.decode(responses[i], skip_special_tokens=True)
            tokens.append(text_to_pieces(sentence, self.tokenizer))

        # 5. 根据熵和 mask 构造高熵 block
        #    注意：如果 build_high_entropy_blocks_tensor 在 CPU 上跑，
        #    entropys / process_mask 要先 .cpu()
        blocks = build_high_entropy_blocks_tensor(
            tokens,
            entropys.cpu(),
            process_mask.cpu(),
            seed_method='mean_std',
            max_block_len=16,
            min_block_len=3,
            stop_on_sentence_boundary=True,
            max_span=128,
            eos_probs=eos_probs,
        )

        # blocks 是一个按 batch 的 object 数组，每个元素是一条样本的若干 block 描述
        data.non_tensor_batch['blocks'] = np.array(blocks, dtype=object)
        return data

    def construct_sft_data_to_update(self, data: DataProto, self_explain_result: DataProto,
                                     error_blocks: list[Optional[tuple | None]]):
        pass

    def gather_self_explain_input(self, data: DataProto, error_blocks: list[Optional[tuple | None]]):
        """
        构造让模型生成下一个步骤的prompt
        :param data: 完整数据，batch size 为 rollout 的次数，每一条 rollout 的数据对应若干个重要的需要重写的 block
        :param error_blocks:
        :return:
        """
        raw_target_prompts = data.non_tensor_batch["raw_tgt_prompts"]
        problems = data.non_tensor_batch["problem"]
        print(f'sampled questions: {problems[0]}')
        responses = data.batch["responses"]
        explain_prompts = []
        answers_prefix = []
        first_incorrect_steps = []
        raw_target_prompts_selected = []
        problems_selected = []
        raw_index = [] # 由于需要se的只有错误的，因此se的输入不一定和原来的尺寸还一样，因此需要记录原始的索引

        for i, block in enumerate(error_blocks):
            if block is None:
                continue
            start, end, index = block
            if start == 0:
                continue
            answer_prefix = self.tokenizer.decode(responses[i][:start])
            assert answer_prefix.strip() != "", \
                f"Answer prefix should not be empty {start}, {self.tokenizer.decode(responses[i])}"
            answers_prefix.append(answer_prefix)
            first_incorrect_steps.append(self.tokenizer.decode(responses[i][start:end], skip_special_tokens=True))
            raw_target_prompts_selected.append(raw_target_prompts[i])
            problems_selected.append(problems[i])
            explain_prompts.append(construct_explain_prompt(problems[i], raw_target_prompts[i], answer_prefix))
            raw_index.append(i)

        # print(f' sampled explain prompts: {explain_prompts[0]}')
        explain_prompts_input_ids = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'] for prompt in
             explain_prompts]
        explain_prompts_attention_mask = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['attention_mask'] for prompt in
             explain_prompts]
        max_input_ids_length = max([input_ids.shape[1] for input_ids in explain_prompts_input_ids])
        explain_prompts_input_ids = [
            pad_sequence_to_length(prompt, max_input_ids_length, self.tokenizer.pad_token_id, left_pad=True)
            for prompt in explain_prompts_input_ids]
        explain_prompts_attention_mask = [
            pad_sequence_to_length(prompt, max_input_ids_length, 0, left_pad=True) for prompt in
            explain_prompts_attention_mask]

        explain_prompts_input_ids = torch.cat(explain_prompts_input_ids, dim=0)
        explain_prompts_attention_mask = torch.cat(explain_prompts_attention_mask, dim=0)
        assert explain_prompts_input_ids.shape == explain_prompts_attention_mask.shape, \
            f"Input ids shape: {explain_prompts_input_ids.shape}, attention mask shape: {explain_prompts_attention_mask.shape}"

        batch = TensorDict(
            {
                "input_ids": explain_prompts_input_ids,
                "attention_mask": explain_prompts_attention_mask,
            },
            batch_size=explain_prompts_input_ids.shape[0],
        )
        non_tensor_batch = {
                "answers_prefix": np.array(answers_prefix, dtype=object),
                "problems": np.array(problems_selected, dtype=object),
                "reference_answers": np.array(raw_target_prompts_selected, dtype=object),
                "first_incorrect_steps": np.array(first_incorrect_steps, dtype=object),
                "raw_index": np.array(raw_index, dtype=object),
            }
        vllm_inputs = DataProto(batch, non_tensor_batch)

        return vllm_inputs

    def gather_sft_blocks_input_embed(self, data: DataProto, error_blocks: list[list[int]]):
        """
        构造让模型生成下一个步骤的prompt
        :param data: 完整数据，batch size 为 rollout 的次数，每一条 rollout 的数据对应若干个重要的需要重写的 block
        :param error_blocks:
        :return:
        """
        prompts_ids = data.batch["prompts"]
        position_ids = data.batch["position_ids"]
        advantages = data.batch["advantages"]
        old_log_probs = data.batch["old_log_probs"]
        raw_target_prompts = data.non_tensor_batch["raw_tgt_prompts"]
        decoded_questions = self.tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
        # print(f'sampled questions: {decoded_questions[0]}')
        decoded_questions = [question.split('User: This is the problem:')[1].split('Assistant:')[0] for question in decoded_questions]
        print(f'sampled questions:')
        for question in decoded_questions[:10]:
            print(question.replace('\n', ' '))

        responses = data.batch["responses"]
        batch_size = data.batch.batch_size[0]
        total_nums = prompts_ids.shape[0]
        print(f'total_nums: {total_nums}')

        new_responses = []
        tgt_responses = []
        new_responses_mask = []
        new_prompts_ids = []
        new_position_ids = []
        new_advantages = []
        new_old_log_probs = []
        answers_prefix = []
        questions = []
        raw_target_prompts_selected = []
        max_blocks_num = self.config.actor_rollout_ref.rollout.self_explain.max_blocks_num
        for i in range(batch_size):
            if len(error_blocks[i]) == 0:
                continue
            # blocks_per_question = error_blocks[i] if len(error_blocks[i]) < max_blocks_num \
            #     else error_blocks[i][:max_blocks_num]
            blocks_per_question = error_blocks[i]
            for block in blocks_per_question:
                start = block[0]
                if start == 0:
                    continue
                end = block[1]
                new_responses.append(responses[i:i+1,:end])
                new_responses_mask_item = torch.zeros_like(responses[i:i+1,...])
                new_responses_mask_item[...,start:end] = 1
                new_responses_mask.append(new_responses_mask_item)

                tgt_responses.append(responses[i:i + 1, :start])

                new_prompts_ids.append(prompts_ids[i:i+1,...])
                new_position_ids.append(position_ids[i:i+1,...])
                new_advantages.append(advantages[i:i+1,...])
                new_old_log_probs.append(old_log_probs[i:i+1,...])
                answers_prefix.append(self.tokenizer.decode(responses[i][:start]))
                questions.append(decoded_questions[i])
                raw_target_prompts_selected.append(raw_target_prompts[i])
                assert answers_prefix[-1].strip() != "", \
                    f"Answer prefix should not be empty {start}, {self.tokenizer.decode(responses[i])}"
        explain_prompts = []
        for i in range(len(questions)):
            explain_prompts.append(construct_explain_prompt(questions[i], raw_target_prompts_selected[i],
                                                            answers_prefix[i]))
        # print(f' explain texts length: {len(explain_prompts)}')
        # print(f' sampled explain prompts: {explain_prompts[0]}')
        # print(f' sampled explain prompts: {explain_prompts[1]}')
        explain_prompts_input_ids = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'] for prompt in
             explain_prompts]
        explain_prompts_attention_mask = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['attention_mask'] for prompt in
             explain_prompts]
        max_input_ids_length = max([input_ids.shape[1] for input_ids in explain_prompts_input_ids])
        # print(f'max input ids length: {max_input_ids_length}')
        explain_prompts_input_ids = [
            pad_sequence_to_length(prompt, max_input_ids_length, self.tokenizer.pad_token_id, left_pad=True)
            for prompt in explain_prompts_input_ids]
        explain_prompts_attention_mask = [
            pad_sequence_to_length(prompt, max_input_ids_length, 0, left_pad=True) for prompt in
            explain_prompts_attention_mask]

        # if len(explain_prompts_input_ids) % self.config.actor_rollout_ref.rollout.chunk_size != 0:
        #     pad_times = self.config.actor_rollout_ref.rollout.chunk_size - len(explain_prompts_input_ids) % \
        #                                                                   self.config.actor_rollout_ref.rollout.chunk_size
        if len(explain_prompts_input_ids) < total_nums:
            pad_times = total_nums - len(explain_prompts_input_ids)
            for i in range(pad_times):
                explain_prompts_input_ids.append(torch.zeros_like(explain_prompts_input_ids[0]))
                explain_prompts_attention_mask.append(torch.zeros_like(explain_prompts_attention_mask[0]))
                new_responses.append(torch.zeros_like(responses[0:1,...]))
                tgt_responses.append(torch.zeros_like(responses[0:1,...]))
                new_responses_mask.append(torch.zeros_like(new_responses_mask[0]))
                new_prompts_ids.append(torch.zeros_like(new_prompts_ids[0]))
                new_advantages.append(torch.zeros_like(new_advantages[0]))
                new_old_log_probs.append(torch.zeros_like(new_old_log_probs[0]))
                answers_prefix.append('')
                raw_target_prompts_selected.append('')
                new_position_ids.append(torch.zeros_like(new_position_ids[0]))
                questions.append('')
        elif len(explain_prompts_input_ids) > total_nums:
            explain_prompts_input_ids = explain_prompts_input_ids[:total_nums]
            explain_prompts_attention_mask = explain_prompts_attention_mask[:total_nums]
            new_responses = new_responses[:total_nums]
            tgt_responses = tgt_responses[:total_nums]
            new_responses_mask = new_responses_mask[:total_nums]
            new_prompts_ids = new_prompts_ids[:total_nums]
            new_advantages = new_advantages[:total_nums]
            new_old_log_probs = new_old_log_probs[:total_nums]
            answers_prefix = answers_prefix[:total_nums]
            raw_target_prompts_selected = raw_target_prompts_selected[:total_nums]
            new_position_ids = new_position_ids[:total_nums]
            questions = questions[:total_nums]
            pad_times = 0
        else:
            pad_times = 0
        explain_prompts_input_ids = torch.cat(explain_prompts_input_ids, dim=0)
        explain_prompts_attention_mask = torch.cat(explain_prompts_attention_mask, dim=0)
        # print(f'input ids: {explain_prompts_input_ids}, shape: {explain_prompts_input_ids.shape}')
        # print(f'attention mask: {explain_prompts_attention_mask}, shape: {explain_prompts_attention_mask.shape}')
        assert explain_prompts_input_ids.shape == explain_prompts_attention_mask.shape, \
            f"Input ids shape: {explain_prompts_input_ids.shape}, attention mask shape: {explain_prompts_attention_mask.shape}"

        batch = TensorDict(
            {
                "input_ids": explain_prompts_input_ids,
                "attention_mask": explain_prompts_attention_mask,
                "response_mask": torch.cat(new_responses_mask, dim=0),
                "prompts": torch.cat(new_prompts_ids, dim=0),
                "position_ids": torch.cat(new_position_ids, dim=0),
                "advantages": torch.cat(new_advantages, dim=0),
                "old_log_probs": torch.cat(new_old_log_probs, dim=0),
            },
            batch_size=explain_prompts_input_ids.shape[0],
        )
        batch["tgt_attention_mask"] = torch.cat([(batch["prompts"] != self.tokenizer.pad_token_id).int()], dim=-1)
        non_tensor_batch = {
                "answers_prefix": np.array(answers_prefix, dtype=object),
                "questions": np.array(questions, dtype=object),
                "raw_target_prompts": np.array(raw_target_prompts_selected, dtype=object),
            }
        vllm_inputs = DataProto(batch, non_tensor_batch)

        return vllm_inputs, new_responses, pad_times, tgt_responses

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # print(f' new batch: {new_batch}')
                num_gen_batches += 1
                # pop those keys for generation
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                )

                # interleave==True: [a, b] -> [a, a, b, b]
                # interleave==False: [a, b] -> [a, b, a, b]
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n-self.config.actor_rollout_ref.rollout.n_off_policy, 
                    interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with ((marked_timer("step", timing_raw))):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output: DataProto = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # print(f'gen batch output: {gen_batch_output}')
                        # 这一部分返回只有五项：
                        # prompts: Tensor(shape=torch.Size([24, 2048]),),
                        # responses: Tensor(shape=torch.Size([24, 20480]),),
                        # input_ids: Tensor(shape=torch.Size([24, 22528]),),
                        # attention_mask: Tensor(shape=torch.Size([24, 22528]),),
                        # position_ids: Tensor(shape=torch.Size([24, 22528]),),
                        n_off_policy = self.config.actor_rollout_ref.rollout.n_off_policy
                        # print(f'gen batch: {gen_batch_output.batch}')
                        if n_off_policy > 0:
                            required_keys = ["prompts", "responses", "input_ids", "attention_mask", "position_ids"]
                            assert all(key in gen_batch_output.batch for key in required_keys), "生成的序列数据缺少必要键"

                            batch_size = new_batch.batch.batch_size[0]

                            # 校验：数据总长度必须是batch_size的整数倍
                            total_samples = gen_batch_output.batch['prompts'].shape[0]
                            assert total_samples % batch_size == 0, f"总样本数{total_samples}不是批次大小{batch_size}的整数倍"
                            gen_batch_output_list = gen_batch_output.batch.split(int(total_samples // batch_size), dim=0)
                            mixed_batch_output_list = []
                            for idx, gen_batch_output_item in enumerate(gen_batch_output_list):
                                off_prompts = gen_batch_output_item['prompts'][-1:,...].clone()
                                off_responses = new_batch.batch['tgt_responses'][idx:idx+1,...].clone()
                                off_input_ids = new_batch.batch['tgt_input_ids'][idx:idx+1,...].clone()
                                off_attention_mask = new_batch.batch['tgt_attention_mask'][idx:idx+1,...].clone()
                                off_position_ids = gen_batch_output_item['position_ids'][-1:,...].clone()
                                off_tensor_dict = TensorDict({
                                    "prompts": off_prompts,
                                    "responses": off_responses,
                                    "input_ids": off_input_ids,
                                    "attention_mask": off_attention_mask,
                                    "position_ids": off_position_ids,
                                }, batch_size=[1])
                                off_tensor_dict: TensorDict = torch.cat([off_tensor_dict] * n_off_policy, dim=0)
                                mixed_batch_output_item: TensorDict = torch.cat([gen_batch_output_item, off_tensor_dict], dim=0)
                                on_policy_mask = torch.ones_like(mixed_batch_output_item['responses'],
                                                                 dtype=torch.int64)
                                on_policy_mask[-n_off_policy:, ...] = 0
                                mixed_batch_output_item['on_policy_mask'] = on_policy_mask
                                mixed_batch_output_list.append(mixed_batch_output_item)
                            gen_batch_output.batch = torch.cat(mixed_batch_output_list, dim=0)
                            gen_batch_mixed_output = gen_batch_output

                        else:                           
                            gen_batch_mixed_output = gen_batch_output
                            on_policy_mask = torch.ones_like(gen_batch_mixed_output.batch['responses'], dtype=torch.int64)
                            gen_batch_mixed_output.batch['on_policy_mask'] = on_policy_mask

                        # print(f'gen batch: {gen_batch_mixed_output.batch}')
                        timing_raw.update(gen_batch_mixed_output.meta_info["timing"])
                        gen_batch_mixed_output.meta_info.pop("timing", None)

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_mixed_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        # print('use rm: ', self.use_rm)
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        reward_tensor, reward_extra_infos_dict = self.reward_fn(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        # if reward_extra_infos_dict:
                        #     new_batch.non_tensor_batch.update(
                        #         {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        #     )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        batch.meta_info.update({"eos_token_id": self.tokenizer.eos_token_id})
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        eos_prob = old_log_prob.batch["eos_prob"]
                        print(f'eos shape: {eos_prob.shape}')
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with marked_timer("blocks_division", timing_raw, "green"):
                        new_batch = self.divide_answers_blocks(new_batch, eos_probs=eos_prob)
                        print(f'finished get_answers_blocks')
                        print(new_batch.non_tensor_batch["blocks"])

                    with marked_timer("error_localization", timing_raw, "yellow"):
                        # error_blocks = filter_blocks_by_embedding(batch.non_tensor_batch["parsed_blocks"].tolist(),
                        #                                              batch.non_tensor_batch["raw_tgt_prompts"].tolist())
                        if self.config.trainer.llm_error_localization:
                            # 使用 LLM 对当前 new_batch 的 parsed_blocks 进行复核与过滤
                            error_blocks, re_verified_true, llm_results = localize_error_by_llm(
                                new_batch.non_tensor_batch["blocks"].tolist(),
                                new_batch.non_tensor_batch["raw_tgt_prompts"].tolist(),
                                new_batch.non_tensor_batch["problem"],
                                reward_extra_infos_dict["acc"])

                            # 对经过 LLM 复核后被判为“整体正确”的样本：
                            # - 在对应的最后一个有效 token 位置打一个 reward = 1.0
                            # - 将 acc 标记为 True（即 LLM 认可为正确）
                            for idx, item in enumerate(re_verified_true):
                                if item:
                                    response_valid_length = reward_extra_infos_dict["response_valid_length"][idx]
                                    reward_tensor[idx][response_valid_length - 1] = 1.0
                                    reward_extra_infos_dict["acc"][idx] = True
                            # 记录每个样本对应的 LLM 判定结果（字典 / None）
                            reward_extra_infos_dict["llm_results"] = llm_results

                        else:
                            error_blocks = self.localize_error_by_emb(new_batch.non_tensor_batch["blocks"].tolist(),
                                new_batch.non_tensor_batch["raw_tgt_prompts"].tolist())
                            print(f'error_blocks: {error_blocks}')

                        # for llm_result, acc in zip(llm_results, reward_extra_infos_dict["acc"]):
                        #     print(acc, '  ', llm_result)

                        # 统计每个样本的正确数
                        acc = [1 if i else 0 for i in reward_extra_infos_dict["acc"]]
                        acc = np.array(acc, dtype=object)
                        uid_list = list(set(new_batch.non_tensor_batch["uid"]))
                        print(f'uid list: {uid_list}')
                        id2correct_num = defaultdict(int)
                        for uid in uid_list:
                            id2correct_num[uid] = acc[new_batch.non_tensor_batch["uid"] == uid].sum()
                        print(f'id2 correct num: {id2correct_num}')

                        # 每一条rollout数据附加一个新的关于正确数的字段
                        reward_extra_infos_dict["correct_num"] = [0] * len(acc)
                        for i in range(len(acc)):
                            reward_extra_infos_dict["correct_num"][i] = id2correct_num[new_batch.non_tensor_batch["uid"][i]]

                        new_batch.non_tensor_batch["error_blocks"] = np.array(error_blocks, dtype=object)

                        new_batch.batch["token_level_rewards"] = reward_tensor
                        # print(reward_tensor.sum(-1))
                        # print(f'error blocks: {error_blocks}')

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v, dtype=object)
                                if isinstance(v, list) else v for k, v in reward_extra_infos_dict.items()}
                            )
                        for k, v in new_batch.non_tensor_batch.items():
                            print(f'{k}: {type(v)}')

                        tmp = new_batch.non_tensor_batch
                        for i in range(len(acc)):
                            print(f'{tmp["acc"][i]} {tmp["llm_results"][i]} {tmp["error_blocks"][i]} {tmp["correct_num"][i]} ')

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"],
                                                                    dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm["norm_adv_by_std_in_grpo"]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    # print(batch.batch['advantages'])
                    # print(f' test_batch: {batch}')
                    # print(f' non tensors batch keys(): {batch.non_tensor_batch.keys()}')
                    # torch.save(batch.batch, os.path.join('entropy_examples', f'{self.global_steps}.pt'))

                    with marked_timer("sft_blocks_prepare", timing_raw, "green"):
                        self_explain_inputs = self.gather_self_explain_input(batch, error_blocks)
                        print(f'self_explain_inputs: {self_explain_inputs}')
                        self_explain_result: DataProto = self.se_rollout_wg.generate_se_blocks(self_explain_inputs)
                        print(f'self_explain_result: {self_explain_result}')

                        self_explain_samples_num = self_explain_result.batch['input_ids'].shape[0]
                        valid_se_examples = []
                        for i in range(self_explain_samples_num):
                            response: torch.Tensor = self_explain_result.batch['responses'][i]
                            response_mask = (response != self.tokenizer.eos_token_id).int() * (response != self.tokenizer.eos_token_id).int()
                            valid_tokens_num = response_mask.sum().item()
                            if valid_tokens_num > 0:
                                valid_se_examples.append(i)
                            print(valid_tokens_num)
                        print(f'valid se examples: {len(valid_se_examples)}, overall samples: {self_explain_samples_num}')
                        print(f'valid examples: {valid_se_examples}')

                        self_explain_result = self_explain_result.select_idxs(valid_se_examples)
                        self_explain_inputs_size = self_explain_result.batch['input_ids'].shape[0]
                        collected = []
                        llm_client = OpenAI(base_url="https://api.vectorengine.ai/v1",
                                            api_key="sk-PqqzpkgeXymtXSLepUSnK9XAuluuyEaRITaXjugJgm22fdwj")
                        corrected_num = 0
                        for i in range(self_explain_inputs_size):
                            self_explain_prompt = self.tokenizer.decode(self_explain_inputs.batch['input_ids'][i],
                                                                  skip_special_tokens=True)
                            # print(f'prefix answer: {prefix_answer}')
                            # print(f'self_explain_prompt: {self_explain_prompt}')
                            # print_tensor(self_explain_result.batch['responses'][i])
                            complete_answer = self.tokenizer.decode(self_explain_result.batch['responses'][i],
                                                                    skip_special_tokens=True) # 模型生成的下一步的答案
                            valid_num = self_explain_result.batch['attention_mask'][i].sum().item()
                            tokens = text_to_pieces(complete_answer, self.tokenizer)
                            _, complete_answer_splits = split_into_blocks(complete_answer, tokens, 192)
                            print(f'complete answer: {complete_answer}')
                            print(f'answer splits: {complete_answer_splits}')
                            print(f'valid num: {valid_num}')
                            answer_prefix = self_explain_inputs.non_tensor_batch['answers_prefix'][i]
                            first_incorrect_step = self_explain_inputs.non_tensor_batch['first_incorrect_steps'][i]
                            # if len(complete_answer_splits) > 0:
                            #     verdict = judge_candidate_step_chat(
                            #         problem=self_explain_inputs.non_tensor_batch['problems'][i],
                            #         prefix_steps=answer_prefix,
                            #         candidate_step=complete_answer_splits[0],
                            #         reference_answer=self_explain_inputs.non_tensor_batch['reference_answers'][i],
                            #         client=llm_client,
                            #     )
                            #     print(verdict)
                            #     if verdict['is_correct']:
                            #         corrected_num += 1
                            # else:
                            #     verdict = {}
                            verdict = {}

                            raw_index = self_explain_result.non_tensor_batch['raw_index'][i]
                            llm_result = batch.non_tensor_batch['llm_results'][raw_index]
                            standard_answer = batch.non_tensor_batch['raw_tgt_prompts'][raw_index]
                            correct_num = batch.non_tensor_batch['correct_num'][raw_index]
                            collected.append({
                                "self_explain_prompt": self_explain_prompt,  # 这个是有self explain的提示prompt的
                                "complete_answer": complete_answer_splits,
                                "completed_tokens_num": valid_num,
                                "answer_prefix": answer_prefix,
                                "problem": self_explain_inputs.non_tensor_batch['problems'][i],
                                "step_incorrect": {
                                    "step": first_incorrect_step,
                                    "eval_result": llm_result
                                },
                                "step_corrected": {
                                    "step": complete_answer_splits[0],
                                    "eval_result": verdict
                                },
                                "answer_incorrect": answer_prefix + first_incorrect_step,
                                "answer_corrected": answer_prefix + complete_answer,
                                "standard_answer": standard_answer,
                                "correct_num": correct_num
                            })
                        corrected_ratio = corrected_num / self_explain_inputs_size
                        print(f'corrected_ratio: {corrected_ratio}')
                        with open(f'self_explain_examples/test/{self.global_steps}.json', 'w', encoding='utf-8') as f:
                            f.write(json.dumps(collected, ensure_ascii=False, indent=4))

                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        print(f'balancing batch: {self.config.trainer.balance_batch}')
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)
                        exit(0)

                        pad_length = sft_blocks_inputs.batch['response_mask'].shape[1]
                        prompt_length = sft_blocks_inputs.batch['prompts'].shape[1]

                        # 用于强化学习的input_ids、attention_mask
                        sft_blocks_inputs.batch['responses'] = torch.cat(
                            [pad_sequence_to_length(response, pad_length, self.tokenizer.pad_token_id, left_pad=False)
                             for response in incomplete_responses], dim=0
                        )
                        sft_blocks_inputs.batch['input_ids'] = torch.cat(
                            [sft_blocks_inputs.batch["prompts"], sft_blocks_inputs.batch['responses']], dim=1
                        )
                        sft_blocks_inputs.batch['attention_mask'] = torch.cat(
                            [sft_blocks_inputs.batch['tgt_attention_mask'], sft_blocks_inputs.batch['response_mask']],
                            dim=-1
                        )

                        sft_blocks_inputs.batch['tgt_responses'] = torch.cat(
                            [pad_sequence_to_length_with_trunc(
                                torch.cat([response, self_explain_result.batch['responses'][idx:idx+1,...]], dim=-1),
                                pad_length, self.tokenizer.pad_token_id, left_pad=False)
                             for idx, response in enumerate(tgt_incomplete_responses)], dim=0
                        )
                        sft_blocks_inputs.batch['tgt_input_ids'] = torch.cat(
                            [sft_blocks_inputs.batch["prompts"], sft_blocks_inputs.batch['tgt_responses']], dim=1
                        )
                        response_tgt_attention_mask = torch.zeros_like(sft_blocks_inputs.batch['response_mask'])
                        for idx, response in enumerate(tgt_incomplete_responses):
                            response_length = response.shape[-1]
                            start = prompt_length+response_length
                            end = start+self_explain_result.batch['attention_mask'][idx].sum().item()
                            response_tgt_attention_mask[idx:idx+1, start:end] = 1
                        sft_blocks_inputs.batch['tgt_attention_mask'] = torch.cat(
                            [sft_blocks_inputs.batch['tgt_attention_mask'], response_tgt_attention_mask], dim=-1
                        )

                        # for idx in range(sft_blocks_inputs.batch['tgt_attention_mask'].shape[0]):
                        #     print('tgt attention mask: ', sft_blocks_inputs.batch['tgt_attention_mask'][idx:idx+1, prompt_length:].sum())
                        #     print('rl attention mask: ', sft_blocks_inputs.batch['attention_mask'][idx:idx+1, prompt_length:].sum())

                    with marked_timer("dynamic sft", timing_raw, "red"):
                        sft_blocks_inputs.meta_info = batch.meta_info
                        sft_blocks_inputs.meta_info['calculate_rl_loss'] = True
                        sft_blocks_inputs.meta_info['calculate_sft_loss'] = True
                        sft_output = self.actor_rollout_wg.update_actor(sft_blocks_inputs)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            # print('batch before update actor ', batch)
                            # print(f' need_analyze_gradients: {self.config.trainer.need_analyze_gradients}')
                            # print(f' save  gradients freq: {self.config.trainer.analyze_gradients_freq}')
                            if self.global_steps % self.config.trainer.get('analyze_gradients_freq', 10) == 0:
                                batch.meta_info['step_index'] = self.global_steps
                                if self.config.trainer.get('need_analyze_sft_grads', False):
                                    batch.meta_info['need_analyze_sft_grads'] = True
                                if self.config.trainer.get('need_analyze_off_grads', False):
                                    batch.meta_info['need_analyze_off_grads'] = True
                            if self.config.actor_rollout_ref.rollout.n_off_policy > 0:
                                batch.meta_info['contain_off_policy'] = True
                            batch.meta_info['calculate_sft_loss'] = False
                            batch.meta_info['calculate_rl_loss'] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def general_reasoner_compute_score(
        self,
        solution_str,
    ):
        solution_slices = solution_str.strip().split('\n')
        final_decision = solution_slices[-1].strip()
        if final_decision == 'Final Decision: Yes':
            return 1
        elif final_decision == 'Final Decision: No':
            return -1
        else:
            return 0

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_kwargs.temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('Validation: Generation end.')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor, _ = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        overall_score = np.mean(reward_tensor.numpy())
        print(f'overall average score: {overall_score:.4f}')
        metric_dict = {f'val/overall_score': overall_score,
                       f'val/overall_correct': np.mean(reward_tensor.numpy()),}
        for data_source, rewards in data_source_reward.items():
            average_score = np.mean(rewards)
            metric_dict[f'val/test_score/{data_source}'] = average_score

        return metric_dict