from collections import defaultdict
from verl import DataProto
import torch

from recipe.mixed_train.utils import grade_answer_mathd, grade_answer_sympy

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    question=None,
    extra_info=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    res = -1.0
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from verl.utils.reward_score import math

        res = math.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from verl.utils.reward_score import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from verl.utils.reward_score import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    else:
        if solution_str == None:
            print("\033[31mthis solution is None!\033[0m")
            res = -1.0
            return res

        # 容错判断：有的ground_truth是字符串，有的ground_truth是列表，最终都转换成列表
        if isinstance(ground_truth, (str, int, float)):
            ground_truth = [ground_truth]
        elif isinstance(ground_truth, list):
            ground_truth = [item for item in ground_truth if item != None]
        else:
            raise NotImplementedError
        ground_truth = [last_boxed_only_string(item) for item in ground_truth]
        if len(ground_truth) == 0:
            res = -1.0
        else:
            for gt in ground_truth:
                if gt == None:
                    continue
                if grade_answer_mathd(solution_str, gt) or grade_answer_sympy(solution_str, gt):
                    res = 1.0
                    break
                else:
                    res = -1.0
    return res

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval.replace('\\boxed{', '')[:-1]

class RuleBasedRewardManager():
    def __init__(
            self, 
            tokenizer, 
            num_examine,
            overlong_buffer_cfg=None,
            max_resp_len=None,
        ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch['rm_scores']
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        for i in range(len(data)): 
            # DataProto虽然有两个成员，只有其中一个成员batch是结构化的数据，但len()返回的就是成员batch的长度
            # get_item返回的是DataProtoItem
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] # prompt是左填充的

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            solution_str = ''
            if '<think>' in response_str and '</think>' in response_str:
                solution_str = response_str.split('</think>')[1]
            if '\\boxed' in response_str:
                solution_str = last_boxed_only_string(response_str)
            if solution_str == '':
                score = -1.0
                # print('response not contain solution: ')
                # print('response: ', response_str)
            else:
                result = compute_score(data_source, solution_str, ground_truth)
                if isinstance(result, float):
                    score = result
                elif isinstance(result, dict):
                    score = result['score']
                else:
                    raise NotImplementedError(f'Un recognized result type: {type(result)}')
                # print('solution str: ', solution_str)
                reward_tensor[i, valid_response_length - 1] = score
            # print('ground truth: ', ground_truth)
            # print('score: ', score)

            reward_extra_info['acc'].append(score == 1.0)
            
            # 进行回答长度过长惩罚
            # if self.overlong_buffer_cfg.enable:
            #     overlong_buffer_len = self.overlong_buffer_cfg.len
            #     expected_len = self.max_resp_len - overlong_buffer_len
            #     exceed_len = valid_response_length - expected_len
            #     overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            #     overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            #     score += overlong_reward
            #     if self.overlong_buffer_cfg.log:
            #         reward_extra_info["overlong_reward"].append(overlong_reward)
            #         reward_extra_info["overlong"].append(overlong_reward < 0)


            # 进行回答打印，只打印num_engine个回答的答案
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor, reward_extra_info
    
if __name__ == "__main__":
    data_source = 'no_source'
    response_str = '<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.'
    ground_truth=["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]
    prompt_str = "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$." 
    if '<think>' in response_str and '</think>' in response_str:
        solution_str = response_str.split('</think>')[1]
    if '\\boxed' in solution_str:
        solution_str = last_boxed_only_string(solution_str)
    
    score = compute_score(data_source, solution_str, ground_truth)
    score = compute_score(data_source, 'x^2 + 2x - 2', 'x^{2}+2x-2')
    print(score)
    test_str = "Thus, the original function's equation is \(\\boxed{y = x^2 + 2x - 2}\)."
    print(last_boxed_only_string(test_str))