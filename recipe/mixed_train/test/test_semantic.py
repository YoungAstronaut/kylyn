import json
from typing import List, Tuple

import torch
from sympy import false
from vllm import LLM

from recipe.dapo.embed_utils import TASK_PREFIX, get_detailed_instruct, TASK, build_embed_inputs, \
    find_first_descent_point, argmin, locate_bad_prefixes, locate_bad_prefix_steps
from recipe.mixed_train.semantic_blocks import split_into_blocks, text_to_pieces
from verl.utils import hf_tokenizer

model_name: str = "../llm_models/Qwen/Qwen3-Embedding-8B"
tokenizer_path = "../llm_models/Qwen/Qwen2.5-7B-Instruct"
model = LLM(model=model_name, task="embed")          # 实际训练循环里可全局复用，不要反复构造

# TASK_PREFIX = (
#     "Given a cumulative reasoning prefix, evaluate whether its semantic similarity "
#     "to the reference solution drops compared to the previous prefix."
# )

def main():
    with open('self_explain_examples/test/1.json', 'r') as f:
        data = json.load(f)

    tokenizer = hf_tokenizer(tokenizer_path)

    false_examples = []
    for item in data:
        steps = item['step_incorrect']['eval_result']['steps']
        incor_step_idx = item['step_incorrect']['eval_result']['k']
        reason = item['step_incorrect']['eval_result']['verdict']['brief_reason']
        standard_answer = item['standard_answer']
        regen_step = item['step_corrected']['step']
        is_regen_step_cor = item['step_corrected']['eval_result'].get('is_correct', false)
        false_examples.append({
            'steps': steps,
            'incor_step_idx': incor_step_idx,
            'standard_answer': standard_answer,
            'regen_step': regen_step,
            'is_regen_step_cor': is_regen_step_cor,
            'reason': reason,
        })

    num = 0
    num_fall = 0
    for example in false_examples:
        print('---------')
        steps = example['steps']
        incor_step_idx = example['incor_step_idx']-1
        standard_answer = example['standard_answer']
        regen_step = example['regen_step']
        # is_regen_step_cor = example['is_regen_step_cor']
        ref_steps = [standard_answer]
        # print(standard_answer)
        standard_answer_tokens = text_to_pieces(standard_answer, tokenizer)
        # print(len(standard_answer_tokens))
        # bad_mask, best_ref_idx, best_sim = locate_bad_prefixes(steps,
        #                     split_into_blocks(standard_answer, standard_answer_tokens, max_span=128)[1])
        print('index of incorrect step: \033[31m', incor_step_idx, '\033[0m \033[33m ', example['reason'], '\033[0m')
        # print('steps')
        # bad_mask, best_ref_idx, best_sim = locate_bad_steps(steps, ref_steps)
        # print('prefixes')
        max_step_per_batch = 10
        cal_times = len(steps) // max_step_per_batch + 1
        loc = -1

        sim_list = []
        for batch_idx in range(cal_times):
            steps_batch = steps[batch_idx * max_step_per_batch:min(len(steps), (batch_idx+1) * max_step_per_batch)]
            bad_mask, best_ref_idx, best_sim = locate_bad_prefixes(steps_batch, ref_steps, model)
            best_sim = best_sim.tolist()
            sim_list += best_sim
            # for i in range(len(best_sim)):
            #     print(f'{i+batch_idx*max_step_per_batch}: {best_sim[i]: .4f}  {[steps[i+batch_idx*max_step_per_batch]]}')
            loc = find_first_descent_point(sim_list)
            if loc != -1:
                loc = loc + batch_idx * max_step_per_batch
                print('loc: ', loc)
                break

        if loc == -1:
            loc = argmin(sim_list)
            print('loc: ', loc)
        for i in range(len(sim_list)):
            if i == loc:
                print(f'{i}: \033[31m{sim_list[i]: .4f}\033[0m  {[steps[i]]}')
            else:
                print(f'{i}: \033[32m{sim_list[i]: .4f}\033[0m  {[steps[i]]}')

        if loc <= incor_step_idx:
            num += 1
            print(True)
        else:
            print(False)

        bad_mask, best_ref_idx, best_sim = locate_bad_prefix_steps(steps, ref_steps)
        print(bad_mask)
        true_indices = torch.nonzero(bad_mask, as_tuple=False)
        print(true_indices)
    print(f'{num / len(false_examples): .4f}')

main()