import torch
import os
from tensordict import TensorDict
from vllm.transformers_utils.tokenizer import encode_tokens

project_name = 'dist_entropy_Qwen2.5-0.5B'
data_path = f'/home/hzchen/jyh/verl/output/dist_entropy/{project_name}'

data_num = 5
data_tensors = []
for i in range(data_num):
    file_path = os.path.join(data_path, f'{i+1}.pt')
    data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    print(type(data))  # 应输出 <class 'tensordict.TensorDict'>
    print(data.keys())
    print(data)
    data_tensors.append(data)
    
from transformers import AutoTokenizer
model_name = '/home/hzchen/jyh/llm_models/Open-Reasoner-Zero/Open-Reasoner-Zero-7B'
tokenizer = AutoTokenizer.from_pretrained(model_name)

HUE_HIGH = 120  # 高概率色调（绿色）
HUE_LOW = 0     # 低概率色调（红色）
SATURATION = 0.7
LIGHTNESS = 0.5

import colorsys
def print_colored_from_ids(token_ids, probs, entropy, skip_tokens=set()):
    """
    根据token ID列表和对应概率，打印带颜色的连贯文本
    参数：
        token_ids: list[int]，token在词表中的ID列表（如[101, 3221, 4567, ...]）
        probs: list[float]，每个token对应的生成概率（长度需与token_ids一致，元素范围0~1）
    """
    # 校验输入长度一致
    if len(token_ids) != len(probs):
        raise ValueError("token_ids和probs的长度必须一致！")
    
    # 1. 将token ID转换为子词（subtoken）
    subtokens = tokenizer.convert_ids_to_tokens(token_ids)
    # 处理特殊token（如[CLS]、[SEP]，可选：过滤或保留）
    # 这里保留特殊token，如需过滤可添加：subtokens = [t for t in subtokens if t not in ['[CLS]', '[SEP]']]
    
    # 2. 生成完整语义文本（用于参考）
    full_text = tokenizer.decode(token_ids, skip_special_tokens=True)  # skip_special_tokens=True：过滤[CLS]等
    # print("完整语义文本（参考）：", full_text)
    print("带概率颜色的子词组合：")
    
    # 3. 为每个子词上色并处理前缀
    colored_parts = []
    average_entropy = sum(entropy) / len(entropy)
    print('average entropy: ', average_entropy)
    for subtoken, prob, token_id, entropy_item in zip(subtokens, probs, token_ids, entropy):
        # 3.1 概率→RGB颜色
        if token_id in skip_tokens:
            continue
        # print('sub token: ', subtoken, ' prob: ', prob, ' token id: ', token_id, ' entropy_item: ', entropy_item)
        prob_clamped = max(0.0, min(entropy_item / 0.3, 1.0))  # 限制概率范围
        hue = HUE_LOW + (HUE_HIGH - HUE_LOW) * prob_clamped
        r, g, b = colorsys.hls_to_rgb(hue / 360, LIGHTNESS, SATURATION)
        r, g, b = int(r*255), int(g*255), int(b*255)
        
        # 3.2 处理子词特殊前缀（确保文本连贯）
        if subtoken.startswith("##"):
            # BERT系：去除##，直接拼接前词
            display_subtoken = subtoken.replace("##", "")
        elif "Ġ" in subtoken:
            # GPT/LLaMA系：Ġ表示空格，替换为空格
            display_subtoken = subtoken.replace("Ġ", " ")
        elif "Ċ" in subtoken:
            display_subtoken = subtoken.replace("Ċ", "\n")
        else:
            # 其他情况（如中文子词、特殊token）：直接使用
            display_subtoken = subtoken
        
        # 3.3 添加ANSI颜色码
        colored_sub = f"\033[38;2;{r};{g};{b}m{display_subtoken}\033[0m"
        colored_parts.append(colored_sub)
    
    # 4. 拼接所有带颜色的子词，打印结果
    print("".join(colored_parts))
    
data_tensor = data_tensors[0]

responses = data_tensor['responses']
inputs_ids = data_tensor['input_ids']
log_probs = data_tensor['rollout_log_probs']
attention_mask = data_tensor['attention_mask']
batch_size = responses.shape[0]
for i in range(batch_size):
    input_ids_single = responses[i].tolist() # 模型输出对应的id
    input_probs_single = []
    input_allprobs_single = []
    entropy_single = []
    
    responses_single = responses[i].tolist()
    log_probs_single = log_probs[i]
    sequence_length = len(responses_single)
    for t in range(sequence_length):
        log_probs_single_index = log_probs_single[t][:,0].to(dtype=torch.int32).tolist()
        probs_single_probs = torch.exp(log_probs_single[t][:,1])
        entropy_single.append(-torch.sum(probs_single_probs * log_probs_single[t][:,1]).item())
        probs_single_probs = probs_single_probs.tolist()
        input_probs_single.append(probs_single_probs[0])
        input_allprobs_single.append({log_probs_single_index[k] : probs_single_probs[k] for k in range(len(log_probs_single_index))})
    
    skip_tokens = set([tokenizer.eos_token_id, tokenizer.pad_token_id])
    result = print_colored_from_ids(input_ids_single, input_probs_single, entropy=entropy_single, skip_tokens=skip_tokens)
    if i >= 10:
        break
    # result = print_colored_from_ids(responses, )