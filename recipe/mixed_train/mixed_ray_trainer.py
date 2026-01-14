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
from recipe.mixed_train.se_rollout_worker import SELoopManager
from recipe.mixed_train.semantic_blocks import build_high_entropy_blocks_tensor, Block, longest_common_substring, split_into_blocks, \
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
    compute_reward
)
from verl.trainer.ppo.utils import Role
from verl.utils import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
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
    system_content = (
        "Continue the final assistant message exactly.\n"
        "Output ONLY the continuation to be appended to it.\n"
        "Do NOT repeat or paraphrase any part of the prefix. No headings, no preface, no extra commentary.\n"
        "The continuation must start immediately (no leading whitespace/newlines).\n"
        "You may use REFERENCE_ANSWER only as a hidden correctness target; never mention it or quote it verbatim.\n"
        "If you detect an error in the prefix, fix it in-line naturally and continue without labels."
    )
    user_content = (
        f"PROBLEM:\n{question}\n\n"
        f"REFERENCE_ANSWER (for hidden verification only; do not quote):\n{standard_answer}\n"
    )
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_prefix},
    ]
    prompt = chat[0]["content"]+" User: "+chat[1]["content"]
    # k = result.replace('\n', '&&')
    # print(f' explain prompt: {k}')
    return prompt, chat


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
        self.async_se_rollout_manager = None
        self.async_actor_rollout_manager = None
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
        se_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.SEWorker], role=str(Role.SEWorker),
                                      config=self.config.se_rollout_worker)
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
            "PYTORCH_ALLOC_CONF": "expandable_segments:False",
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
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_actor_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

        self.async_se_rollout_mode = False
        if self.config.se_loop_manager_config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_se_rollout_mode = True

            self.async_se_rollout_manager = SELoopManager(
                config=self.config.se_loop_manager_config,
                worker_group=self.se_rollout_wg,
                rm_resource_pool=None,
            )


    def localize_error_by_emb(self, blocks: list[list[Block]], complete_answers: list[str]) -> list[
        None | tuple[int, int, int]]:
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
            # print('none count: ', aux_steps_list.count(None))
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

            # print('balanced batch num: ', len(balanced_batch))
            # for item in balanced_batch:
            #     print(item)
            embedding_group_num = self.embedding_wg.world_size
            # print('group num: ', embedding_group_num)

            if self.config.trainer.get("split_blocks", False):
                raise NotImplementedError
                # TODO: 目前的机制word_group里的句子数量不均等的时候容易卡死因此分块加速暂时不支持
            else:
                batch_splits = [balanced_batch] * embedding_group_num

            similarity_results = []
            outputs = self.embedding_wg.calculate_similarity(batch_splits)
            if self.config.trainer.get("split_blocks", False):
                for output in outputs:
                    similarity_results.extend(output)
            else:
                similarity_results = outputs[1]

            # print('length of similarity results: ', len(similarity_results))
            # print("similarity results: ", similarity_results)
            # print('empty similarity results: ', similarity_results.count({"scores": []}))

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
            # error_blocks.append(block_list[err_idx_list[i]])
            get_block = block_list[err_idx_list[i]]
            error_blocks.append((get_block.start, get_block.end, err_idx_list[i]))
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
        raw_index_list = self_explain_result.non_tensor_batch["raw_index"].tolist()
        prompts = data.batch["prompts"].clone()
        attention_mask = data.batch["attention_mask"].clone()

        responses = data.batch["responses"].clone()
        pad_id = self.tokenizer.pad_token_id
        responses_length = responses.shape[-1]

        zero_responses_mask = torch.zeros_like(responses)
        ones_responses_mask = torch.ones_like(responses)
        new_responses = self_explain_result.non_tensor_batch["raw_responses"].tolist()

        tgt_responses_list: list[Optional[torch.Tensor]] = [None] * data.batch.batch_size[0]
        tgt_loss_mask_list: list[Optional[torch.Tensor]] = [None] * data.batch.batch_size[0]

        count = 0
        for i, raw_index in enumerate(raw_index_list):
            if new_responses[i] == "":
                continue
            start, end, _ = error_blocks[raw_index]
            block_ids = self.tokenizer(new_responses[i], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            # print("block_ids", block_ids.shape)
            block_attn_mask = torch.ones_like(block_ids)
            tgt_response = torch.cat([responses[raw_index][:start], block_ids], dim=-1)
            tgt_loss_mask = torch.cat([zero_responses_mask[raw_index][:start], block_attn_mask], dim=-1)

            tgt_responses_list[raw_index] = tgt_response
            tgt_loss_mask_list[raw_index] = tgt_loss_mask
            count += 1
        print("count ", count)

        for idx in range(len(tgt_responses_list)):
            if tgt_responses_list[idx] is None:
                tgt_responses_list[idx] = responses[idx].clone()
                tgt_loss_mask_list[idx] = zero_responses_mask[idx].clone()
            else:
                tgt_responses_list[idx] = pad_sequence_to_length(tgt_responses_list[idx], responses_length, pad_id,
                                                                 left_pad=False)
                tgt_loss_mask_list[idx] = pad_sequence_to_length(tgt_loss_mask_list[idx], responses_length, 0,
                                                                 left_pad=False)

        tgt_responses = torch.stack(tgt_responses_list, dim=0)
        tgt_response_mask = torch.stack(tgt_loss_mask_list, dim=0)

        tgt_resp_attn_mask = (tgt_responses != pad_id).long()

        tgt_input_ids = torch.cat([prompts, tgt_responses], dim=-1)
        prompt_length = prompts.shape[-1]
        tgt_attention_mask = torch.cat([attention_mask[:, :prompt_length], tgt_resp_attn_mask], dim=-1)
        tgt_position_ids = compute_position_id_with_mask(tgt_attention_mask)

        return DataProto.from_dict({
            "tgt_input_ids": tgt_input_ids,
            "tgt_attention_mask": tgt_attention_mask,
            "tgt_responses": tgt_responses,
            "tgt_response_mask": tgt_response_mask,
            "tgt_position_ids": tgt_position_ids,
        })

    def gather_self_explain_input(self, data: DataProto, error_blocks: list[Optional[tuple | None]],
                                  error_type: list[str]):
        """
        构造让模型生成下一个步骤的prompt
        :param error_type:
        :param data: 完整数据，batch size 为 rollout 的次数，每一条 rollout 的数据对应若干个重要的需要重写的 block
        :param error_blocks:
        :return:
        """
        raw_target_prompts = data.non_tensor_batch["raw_tgt_prompts"]
        problems = data.non_tensor_batch["problem"]
        responses = data.batch["responses"]
        explain_prompts = []
        explain_prompts_chat = []
        answers_prefix = []
        first_incorrect_steps = []
        raw_target_prompts_selected = []
        problems_selected = []
        raw_index = [] # 由于需要se的只有错误的，因此se的输入不一定和原来的尺寸还一样，因此需要记录原始的索引

        for i, block in enumerate(error_blocks):
            if block is None:
                continue
            start, end, index = block
            if start == 0 or error_type[i] == "format_error":
                continue
            answer_prefix = self.tokenizer.decode(responses[i][:start])
            assert answer_prefix.strip() != "", \
                f"Answer prefix should not be empty {start}, {self.tokenizer.decode(responses[i])}"
            answers_prefix.append(answer_prefix)
            first_incorrect_steps.append(self.tokenizer.decode(responses[i][start:end], skip_special_tokens=True))
            raw_target_prompts_selected.append(raw_target_prompts[i])
            problems_selected.append(problems[i])
            explain_prompt, explain_chat = construct_explain_prompt(problems[i], raw_target_prompts[i], answer_prefix)
            explain_prompts.append(explain_prompt)
            explain_prompts_chat.append(explain_chat)
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
            "raw_prompt": np.array(explain_prompts_chat, dtype=list)
        }
        vllm_inputs = DataProto(batch, non_tensor_batch)

        return vllm_inputs

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
        # self.total_training_steps = 40
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
                # print(f"new batch: {new_batch.batch}")
                # print(f"non tensor keys: ", new_batch.non_tensor_batch.keys())
                num_gen_batches += 1
                # pop those keys for generation
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt", "data_source", "reward_model", "ability", "extra_info"]
                )

                # interleave==True: [a, b] -> [a, a, b, b]
                # interleave==False: [a, b] -> [a, b, a, b]
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n-self.config.actor_rollout_ref.rollout.n_off_policy, 
                    interleave=True
                )

                import copy
                gen_batch_copy = copy.deepcopy(gen_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with ((marked_timer("step", timing_raw))):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output: DataProto = self.async_actor_rollout_manager.generate_sequences(gen_batch)
                        # print(f'gen batch output: {gen_batch_output}')
                        reward_extra_info = gen_batch_output.non_tensor_batch["reward_extra_info"]
                        correct_num = 0
                        for extra_info in reward_extra_info:
                            if extra_info['score'] > 0:
                                correct_num += 1
                            # print(f"score: {extra_info['score']}, solution str: {extra_info['solution_str']}, "
                            #       f"ground truth: {extra_info['ground_truth']}")
                        pre_correct_ratio = correct_num / len(reward_extra_info)
                        # print("pre correct ratio: ", pre_correct_ratio)
                        metrics.update({"critic/acc/pre": pre_correct_ratio})
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
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        # print(f'reward_extra_infos_dict: ', reward_extra_infos_dict)

                        new_batch.batch["token_level_scores"] = reward_tensor

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

                    batch = new_batch
                    if "response_mask" in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode",
                                                                                                  False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            batch.meta_info.update({"eos_token_id": self.tokenizer.eos_token_id})
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            eos_prob = old_log_prob.batch["eos_prob"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            # old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    with marked_timer("blocks_division", timing_raw, "green"):
                        new_batch = self.divide_answers_blocks(new_batch, eos_probs=eos_prob)
                        steps_list = []
                        for steps in new_batch.non_tensor_batch["blocks"].tolist():
                            steps_list.append([step.text for step in steps])

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
                            # print(f'error_blocks: {error_blocks}')

                        # for llm_result, acc in zip(llm_results, reward_extra_infos_dict["acc"]):
                        #     print(acc, '  ', llm_result)

                        # 统计每个样本的正确数
                        # reward_extra_infos_dict["score"] 的格式是[-1,-1,...1]
                        score = reward_extra_infos_dict["score"]
                        score = np.where(score == -1, 0, score)
                        # print('score, ', score)
                        uid_list = list(set(new_batch.non_tensor_batch["uid"]))
                        id2correct_num = defaultdict(int)
                        for uid in uid_list:
                            id2correct_num[uid] = score[new_batch.non_tensor_batch["uid"] == uid].sum()

                        # 每一条rollout数据附加一个新的关于正确数的字段
                        reward_extra_infos_dict["correct_num"] = [0] * len(score)
                        for i in range(len(score)):
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
                        # for k, v in new_batch.non_tensor_batch.items():
                            # print(f'{k}: {type(v)}')

                        # tmp = new_batch.non_tensor_batch
                        # for i in range(len(score)):
                        #     print(f'{tmp["score"][i]} {tmp["error_blocks"][i]} {tmp["correct_num"][i]} ')

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, "brown"):
                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

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

                    sft_indexes = {}
                    with marked_timer("sft_blocks_prepare", timing_raw, "green"):
                        error_type = batch.non_tensor_batch["error_type"]
                        self_explain_inputs = self.gather_self_explain_input(batch, error_blocks, error_type)
                        self_explain_samples_num = self_explain_inputs.batch.batch_size[0]
                        self_explain_inputs.pop(batch_keys=["input_ids", "attention_mask"])
                        self_explain_result: DataProto = self.async_se_rollout_manager.generate_sequences(self_explain_inputs)
                        self_explain_inputs.pop(non_tensor_batch_keys=["raw_prompt"])
                        self_explain_result = self_explain_result.union(self_explain_inputs)
                        print("self explain: ", self_explain_result.batch.batch_size[0])

                        collected = []
                        raw_responses = []
                        for i in range(self_explain_samples_num):
                            answer_prefix = self_explain_inputs.non_tensor_batch['answers_prefix'][i]
                            first_incorrect_step = self_explain_inputs.non_tensor_batch['first_incorrect_steps'][i]
                            self_explain_prompt = self.tokenizer.decode(self_explain_result.batch['input_ids'][i],
                                                                  skip_special_tokens=True)
                            complete_answer = self.tokenizer.decode(self_explain_result.batch['responses'][i],
                                                                    skip_special_tokens=True)
                            if answer_prefix.strip() in complete_answer: # 判断模型是否重复了一遍前缀
                                # print("answer prefix: ", [answer_prefix], "complete answer: ", [complete_answer])
                                complete_answer = complete_answer.replace(answer_prefix, "").strip()
                            if len(longest_common_substring(complete_answer, answer_prefix)) > 50:
                                # print("answer prefix: ", [answer_prefix], "complete answer: ", [complete_answer], "common part: ", [longest_common_substring(complete_answer, answer_prefix)])
                                complete_answer = ""
                            if "\\boxed" in answer_prefix:
                                # print("answer already in answer prefix", [answer_prefix])
                                complete_answer = ""
                            
                            raw_responses.append(complete_answer)
                            valid_num = self_explain_result.batch["response_mask"][i].sum().item()

                            raw_index = self_explain_inputs.non_tensor_batch["raw_index"][i]
                            if raw_index in sft_indexes.keys():
                                sft_indexes[raw_index] += 1
                            else:
                                sft_indexes[raw_index] = 1
                            standard_answer = batch.non_tensor_batch['raw_tgt_prompts'][raw_index]
                            correct_num = batch.non_tensor_batch['correct_num'][raw_index]
                            if complete_answer != "":
                                collected.append({
                                    "self_explain_prompt": self_explain_prompt,  # 这个是有self explain的提示prompt的
                                    "incorrect_step": first_incorrect_step,
                                    "complete_answer": complete_answer,
                                    "completed_tokens_num": valid_num,
                                    "answer_prefix": answer_prefix,
                                    "problem": self_explain_inputs.non_tensor_batch['problems'][i],
                                    "standard_answer": standard_answer,
                                    "correct_num": correct_num,
                                    "steps": steps_list[raw_index]
                                })
                        if not os.path.exists(f'self_explain_examples/{self.config.trainer.experiment_name}'):
                            os.makedirs(f'self_explain_examples/{self.config.trainer.experiment_name}')
                        with open(f'self_explain_examples/{self.config.trainer.experiment_name}/{self.global_steps}.json', 'w', encoding='utf-8') as f:
                            f.write(json.dumps(collected, ensure_ascii=False, indent=4))
                        if self.global_steps <= self.config.trainer.pure_rl_steps:
                            raw_responses = ["" for _ in range(len(raw_responses))]
                        self_explain_result.non_tensor_batch["raw_responses"] = np.array(raw_responses)

                        sft_data_to_update = self.construct_sft_data_to_update(batch, self_explain_result, error_blocks)
                        # print(f"sft_data_to_update: {sft_data_to_update}")
                        batch.pop(batch_keys=["tgt_input_ids","tgt_attention_mask","tgt_responses"])
                        batch = batch.union(sft_data_to_update)
                        # tgt_responses = sft_data_to_update.batch["tgt_responses"]
                        # tgt_response_mask = sft_data_to_update.batch["tgt_response_mask"]
                        # batch_size = tgt_responses.shape[0]
                        # for i in range(batch_size):
                        #     sentence_tensor = tgt_responses[i][tgt_response_mask[i] == 1]
                        #     sentence = self.tokenizer.decode(sentence_tensor, skip_special_tokens=True)
                        #     print(f"tgt_response: {sentence}")

                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        print(f'balancing batch: {self.config.trainer.balance_batch}')
                        # if self.config.trainer.balance_batch:
                        #     self._balance_batch(batch, metrics=metrics)

                    # update actor
                    responses = batch.batch["responses"]
                    response_mask = batch.batch["response_mask"]
                    tgt_responses = batch.batch["tgt_responses"]
                    tgt_response_mask = batch.batch["tgt_response_mask"]
                    # print("batch: ", batch.batch)
                    # for i in range(responses.shape[0]):
                    #     print("RL: ", self.tokenizer.decode(responses[i][tgt_response_mask[i] == 1]))
                    #     print("SFT: ", self.tokenizer.decode(tgt_responses[i][tgt_response_mask[i] == 1]))
                    with marked_timer("update_actor", timing_raw, "red"):
                        batch.meta_info['calculate_sft_loss'] = True
                        batch.meta_info['calculate_rl_loss'] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                    # validate
                    # if (
                    #     self.val_reward_fn is not None
                    #     and self.config.trainer.test_freq > 0
                    #     and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    # ):
                    #     with marked_timer("testing", timing_raw, "green"):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)
                    #
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