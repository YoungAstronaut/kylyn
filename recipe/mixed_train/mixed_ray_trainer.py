"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import uuid
from collections import defaultdict
from enum import Enum
from idlelib.rpc import response_queue
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import Sampler
from tqdm import tqdm

from recipe.mixed_train.semantic_blocks import build_high_entropy_blocks_tensor
from verl import DataProto
from verl.protocol import unpad_dataproto, pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, AdvantageEstimator
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
from verl.utils import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.torch_functional import pad_sequence_to_length, get_response_mask
from verl.utils.tracking import ValidationGenerationsLogger


class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    AnswersChecker = 7

TASK = ('Given a query of a math question and the standard answer, '
         'conclude whether the query is related to the standard answer.')

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

class RayMixedTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
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
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

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
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 新添加的内容
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.AnswersChecker)
        ac_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.AnswersChecker], config=self.config.answers_checker)
        ac_runtime_env = {
            "runtime_env": {
                "env_vars": {
                    "VLLM_USE_V1": "0",  # embed task needs V0
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        }
        ac_cls.update_options(ac_runtime_env)

        self.resource_pool_to_cls[resource_pool]["ac"] = ac_cls

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
            assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        self.answers_checker_wg = all_wg["ac"]
        self.answers_checker_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def get_answers_blocks(self, data: DataProto):
        responses = data.batch['responses']
        advantages = data.batch['advantages']
        entropys = data.batch['entropys']
        response_mask = data.batch['response_mask']
        process_mask = (advantages < 0).int() * response_mask
        flat_responses = responses.view(-1).tolist()
        flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_responses)
        flat_tokens = [token.replace("Ġ", " ").replace("Ċ", "\n") if token is not None else "" for token in flat_tokens]
        tokens = [flat_tokens[i * responses.size(1):(i + 1) * response_mask.size(1)] for i in range(responses.size(0))]
        blocks = build_high_entropy_blocks_tensor(tokens, entropys, process_mask * response_mask, seed_method='mean_std',
                                                  max_block_len=16, min_block_len=3, stop_on_sentence_boundary=True)
        data.non_tensor_batch['parsed_blocks'] = np.array(blocks, dtype=object)
        queries = []
        blocks_length = []
        for block_list in blocks:
            blocks_length.append(len(block_list))
            query_list = []
            for block in block_list:
                query_list.append(get_detailed_instruct(task_description=TASK, query=block.text))
            queries.append(query_list)
        data.non_tensor_batch['blocks_length'] = np.array(blocks_length)
        data.non_tensor_batch['queries'] = np.array(queries, dtype=object)
        return data

    def construct_explain_prompt(self, question: str, standard_answer: str, answer_prefix: str):
        chat = [
            {
                "content": f"Your task is to understand a given standard problem solving process of a given question, "
                           f"then finish an incomplete reasoning process. The question is {question} The standard "
                           f"solving process is as followings: \n{standard_answer}",
                "role": "system"
            },
            {
                "content": f"Finish the following incomplete answer: \n{answer_prefix}",
                "role": "user"
            }
        ]
        return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    def gather_sft_blocks_input(self, prompts: DataProto):
        prompts_ids = prompts.batch["prompts"]
        position_ids = prompts.batch["position_ids"]
        advantages = prompts.batch["advantages"]
        old_log_probs = prompts.batch["old_log_probs"]
        raw_target_prompts = prompts.non_tensor_batch["raw_tgt_prompts"]
        decoded_questions = self.tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
        print(f'sampled questions: {decoded_questions[0]}')
        decoded_questions = [question.split('User:')[1].split('Assistant:')[0] for question in decoded_questions]
        print(f'sampled questions: {decoded_questions[0]}')

        responses = prompts.batch["responses"]
        filtered_blocks = prompts.non_tensor_batch["filtered_blocks"]
        batch_size = prompts.batch.batch_size[0]
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
            if len(filtered_blocks[i]) == 0:
                continue
            blocks_per_question = filtered_blocks[i] if len(filtered_blocks[i]) < max_blocks_num \
                else filtered_blocks[i][:max_blocks_num]
            for block in blocks_per_question:
                start = block[0]
                if start == 0:
                    continue
                end = block[1]
                # print(f'start: {start}, end: {end}')
                new_responses.append(responses[i:i+1,:end])
                new_responses_mask_item = torch.zeros_like(responses[i:i+1,...])
                new_responses_mask_item[...,start:end] = 1
                new_responses_mask.append(new_responses_mask_item)

                tgt_responses.append(responses[i:i + 1, :start])
                # print(f' new responses: {new_responses[-1].shape}')
                # print(f' new tgt responses: {tgt_responses[-1].shape}')

                new_prompts_ids.append(prompts_ids[i:i+1,...])
                new_position_ids.append(position_ids[i:i+1,...])
                new_advantages.append(advantages[i:i+1,...])
                new_old_log_probs.append(old_log_probs[i:i+1,...])
                answers_prefix.append(self.tokenizer.decode(responses[i][:start]))
                questions.append(decoded_questions[i])
                raw_target_prompts_selected.append(raw_target_prompts[i])
                assert answers_prefix[-1].strip() != "", \
                    f"Answer prefix should not be empty {start}, {self.tokenizer.decode(responses[i])}"
                # print(f' answers prefix: {answers_prefix[-1]}')
        explain_prompts = []
        for i in range(len(questions)):
            explain_prompts.append(self.construct_explain_prompt(questions[i], raw_target_prompts_selected[i],
                                                                 answers_prefix[i]))
        print(f' explain texts length: {len(explain_prompts)}')
        print(f' sampled explain prompts: {explain_prompts[0]}')
        # print(f' sampled explain prompts: {explain_prompts[1]}')
        explain_prompts_input_ids = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'] for prompt in
             explain_prompts]
        explain_prompts_attention_mask = \
            [self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['attention_mask'] for prompt in
             explain_prompts]
        max_input_ids_length = max([input_ids.shape[1] for input_ids in explain_prompts_input_ids])
        print(f'max input ids length: {max_input_ids_length}')
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

                with (marked_timer("step", timing_raw)):
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
                                off_tensor_dict = torch.cat([off_tensor_dict] * n_off_policy, dim=0)
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
                        print('use rm: ', self.use_rm)
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_tensor, reward_extra_infos_dict = self.reward_fn(new_batch)
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

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

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        # TODO: 需要确认这里是否需要std归一化
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    # print(f' test_batch: {batch}')
                    # print(f' non tensors batch keys(): {batch.non_tensor_batch.keys()}')
                    # torch.save(batch.batch, os.path.join('entropy_examples', f'{self.global_steps}.pt'))

                    # # TODO：在外部处理语义块分割
                    # batch = self.get_answers_blocks(batch)
                    # # print(f'batch with parsed blocks: {batch}')
                    #
                    # batch = self.answers_checker_wg.generate_answers_mask(batch)
                    # print(f'type of batch: {type(batch)}, type of item in batch: {type(batch[0])}')
                    # # print(f' batch: {batch.batch}')
                    #
                    # sft_blocks_inputs, incomplete_responses, pad_times, tgt_incomplete_responses = \
                    #                 self.gather_sft_blocks_input(batch)
                    # print(
                    #     f'input: {self.tokenizer.decode(sft_blocks_inputs.batch["input_ids"][0], skip_special_tokens=True)}')
                    # sft_inputs_batch = sft_blocks_inputs.pop(
                    #     batch_keys=["input_ids", "attention_mask"]
                    # )
                    # # print(f' sft blocks input: {sft_blocks_inputs}')
                    # sft_blocks_result = self.actor_rollout_wg.generate_sft_blocks(sft_inputs_batch)
                    # print(f'complete responses: {self.tokenizer.decode(sft_blocks_result.batch["responses"][0], skip_special_tokens=True)}')
                    # # print(f'sft_blocks_result: {sft_blocks_result}')
                    # pad_length = sft_blocks_inputs.batch['response_mask'].shape[1]
                    # prompt_length = sft_blocks_inputs.batch['prompts'].shape[1]
                    #
                    # # 用于强化学习的input_ids、attention_mask
                    # sft_blocks_inputs.batch['responses'] = torch.cat(
                    #     [pad_sequence_to_length(response, pad_length, self.tokenizer.pad_token_id, left_pad=False)
                    #      for response in incomplete_responses], dim=0
                    # )
                    # sft_blocks_inputs.batch['input_ids'] = torch.cat(
                    #     [sft_blocks_inputs.batch["prompts"], sft_blocks_inputs.batch['responses']], dim=1
                    # )
                    # sft_blocks_inputs.batch['attention_mask'] = torch.cat(
                    #     [sft_blocks_inputs.batch['tgt_attention_mask'], sft_blocks_inputs.batch['response_mask']],
                    #     dim=-1
                    # )
                    #
                    # sft_blocks_inputs.batch['tgt_responses'] = torch.cat(
                    #     [pad_sequence_to_length(
                    #         torch.cat([response, sft_blocks_result.batch['responses'][idx:idx+1,...]], dim=-1),
                    #         pad_length, self.tokenizer.pad_token_id, left_pad=False)
                    #      for idx, response in enumerate(tgt_incomplete_responses)], dim=0
                    # )
                    # sft_blocks_inputs.batch['tgt_input_ids'] = torch.cat(
                    #     [sft_blocks_inputs.batch["prompts"], sft_blocks_inputs.batch['tgt_responses']], dim=1
                    # )
                    # pad_tgt_attention_mask = torch.zeros_like(sft_blocks_inputs.batch['response_mask'])
                    # for idx, response in enumerate(tgt_incomplete_responses):
                    #     response_length = response.shape[-1]
                    #     start = prompt_length+response_length
                    #     end = start+sft_blocks_result.batch['responses'].shape[1]
                    #     pad_tgt_attention_mask[idx:idx+1, start:end] = 1
                    # sft_blocks_inputs.batch['tgt_attention_mask'] = torch.cat(
                    #     [sft_blocks_inputs.batch['tgt_attention_mask'], pad_tgt_attention_mask], dim=-1
                    # )
                    # # for idx in range(sft_blocks_inputs.batch['tgt_attention_mask'].shape[0]):
                    # #     print((sft_blocks_inputs.batch['tgt_input_ids'][idx:idx+1, prompt_length:]!=self.tokenizer.pad_token_id).sum())
                    # #     print(sft_blocks_inputs.batch['tgt_attention_mask'][idx:idx+1, prompt_length:].sum())
                    # #     print((sft_blocks_inputs.batch['input_ids'][idx:idx + 1, prompt_length:]!=self.tokenizer.pad_token_id).sum())
                    # #     print(sft_blocks_inputs.batch['attention_mask'][idx:idx+1, prompt_length:].sum())
                    # # print(f' sft blocks input: {sft_blocks_inputs}')
                    #
                    # with marked_timer("dynamic sft", timing_raw, "red"):
                    #     sft_blocks_inputs.meta_info = batch.meta_info
                    #     sft_blocks_inputs.meta_info['calculate_rl_loss'] = True
                    #     sft_blocks_inputs.meta_info['calculate_sft_loss'] = True
                    #     sft_output = self.actor_rollout_wg.update_actor(sft_blocks_inputs)

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
                            batch.meta_info['calculate_sft_loss'] = self.config.actor_rollout_ref.actor.calculate_sft_loss
                            batch.meta_info['calculate_rl_loss'] = self.config.actor_rollout_ref.actor.calculate_rl_loss
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