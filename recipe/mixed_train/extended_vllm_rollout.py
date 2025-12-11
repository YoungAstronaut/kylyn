import logging
import os
import numpy as np

import torch
from tensordict import TensorDict
from vllm import SamplingParams

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class ExtendedVLLMRollout(vLLMRollout):
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_se_blocks(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        vllm_inputs = [
            {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
        ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            new_sampling_params = {'n': self.sampling_params.n,
                                   'logprobs': self.sampling_params.logprobs,
                                   'max_tokens': 200,
                                   'detokenize': False,
                                   'temperature': self.sampling_params.temperature,
                                   'top_p': self.sampling_params.top_p,
                                   'top_k': self.sampling_params.top_k,
                                   'ignore_eos': self.sampling_params.ignore_eos}
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already converted it to prompt token id
                sampling_params=SamplingParams(**new_sampling_params),
                use_tqdm=False,
            )

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=200).to(
                idx.device
            )

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )

        batch = TensorDict(
            {
                "responses": response,
                "attention_mask": response_attention_mask,
                "input_ids": idx,
                "input_attention_mask": attention_mask,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

