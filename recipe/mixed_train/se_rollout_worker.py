import logging
import os

from omegaconf import DictConfig
from ray.actor import ActorHandle
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.ray import RayWorkerGroup, RayResourcePool
from verl.trainer.ppo.utils import Role
from verl.utils.device import (
    get_device_name, get_device_id, get_torch_device,
)
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica, SGLangHttpServer, \
    SGLangHttpServerBase
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter
from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

import os, asyncio

import ray

@ray.remote(num_cpus=1)
class SESGLangHttpServer(SGLangHttpServerBase):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        worker_key=str(Role.SEWorker)
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank,
                         node_rank, nnodes, cuda_visible_devices)
        self.worker_key = worker_key

    async def sleep(self):
        await asyncio.gather(*[worker.seworker_sleep.remote() for worker in self.workers])

    async def wake_up(self):
        await asyncio.gather(*[worker.seworker_wake_up.remote() for worker in self.workers])


class SESGLangReplica(SGLangReplica):
    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            print('got workers: ', workers)
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"se_sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = SESGLangHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

class SELoopManager(AgentLoopManager):
    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        self.rollout_replica_class = SESGLangReplica
        assert worker_group is not None
        super().__init__(config, worker_group, rm_resource_pool)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"se_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

class SEServerAdapter(ServerAdapter):
    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"se_sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path, host=host, port=server_port, launch_server=False
        )

def pack_se_reults(input_prompts: DataProto, output: DataProto):
    batch = TensorDict(
        {
            "responses": output.batch["responses"],
            "attention_mask": output.batch["response_mask"],
            "input_ids": input_prompts.batch["input_ids"],
            "input_attention_mask": input_prompts.batch["attention_mask"],
        }
    )
    return DataProto(batch=batch, non_tensor_batch=output.non_tensor_batch)

class SERolloutWorker(AsyncActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, 'rollout', **kwargs)
        print('caught role: ', role)
        from verl.workers.rollout.base import _ROLLOUT_REGISTRY
        _ROLLOUT_REGISTRY[("se_sglang", "async")] = "recipe.mixed_train.se_rollout_worker.SEServerAdapter"
        print('finished initializing se rollout worker')
        print(f'is actor: {self._is_actor}, is rollout: {self._is_rollout}, is ref: {self._is_ref}')

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        print("generating self explain sequences")
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output