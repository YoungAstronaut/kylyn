"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.mixed_train.answers_checker import AnswersChecker
from verl.utils.device import is_cuda_available

from .mixed_ray_trainer import RayMixedTrainer, Role

@hydra.main(config_path="config", config_name="mixed_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import CriticWorker
            from recipe.mixed_train.mixed_train_actor_ref_worker import MixedTrainActorRefWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import CriticWorker
            from recipe.mixed_train.mixed_train_actor_ref_worker import MixedTrainActorRefWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MixedTrainActorRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.AnswersChecker: ray.remote(AnswersChecker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.AnswersChecker: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from recipe.dist_entropy.GeneralVerifierWorker import GeneralVerifierWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(GeneralVerifierWorker)
            mapping[Role.RewardModel] = global_pool_id
        else:
            print('reward model is disabled')

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(MixedTrainActorRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Note(haibin.lin): please make sure custom reward managers are imported and
        # registered via `verl.workers.reward_manager.register`
        from .math_reward import RuleBasedRewardManager
        reward_fn = RuleBasedRewardManager(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RuleBasedRewardManager(tokenizer=tokenizer, num_examine=1)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from recipe.mixed_train.rl_dataset_with_tgt import RLHFDatasetWithTarget
        train_dataset = RLHFDatasetWithTarget(parquet_files=config.data.train_files,
            tokenizer=tokenizer,
            prompt_key=config.data.prompt_key,
            max_prompt_length=config.data.max_prompt_length,
            filter_prompts=config.data.filter_prompts, return_raw_chat=config.data.return_raw_chat,
            truncation=config.data.truncation,
            max_target_length=config.data.max_target_length,
            filter_targets=config.data.filter_targets,
            sample_target_ratio=config.data.sample_target_ratio,
        )
        trainer = RayMixedTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
            train_dataset=train_dataset
        )
        trainer.init_workers() # 在这个函数里会进行模型（各worker）的初始化
        print('\033[32mInitialized workers\033[0m')
        trainer.fit()


if __name__ == "__main__":
    main()
