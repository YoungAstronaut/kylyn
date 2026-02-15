import json
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
MODEL_NAME = "/home/hzchen/jyh/llm_models/Elliott/LUFFY-Qwen-Math-7B-Zero" 
TENSOR_PARALLEL_SIZE = 1
INPUT_FILE = "collected.json"
OUTPUT_FILE = "inference_results_single.json"
# ===========================================

def load_data(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data)} 条数据。")
    return data

def main():
    data = load_data(INPUT_FILE)
    data = data[:500]
    
    lengths = []
    for item in data:
        lengths.append(item["response length"])
    print("平均初始长度：", sum(lengths) / len(lengths))    

    print(f"正在加载模型: {MODEL_NAME} (单卡模式) ...")
    llm = LLM(
        model=MODEL_NAME, 
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    # --- 修改点：直接生成文本 Prompt，避开 prompt_token_ids 参数兼容性问题 ---
    prompts_text = []
    print("正在处理 Chat Template ...")
    for item in data:
        messages = item['prompt']
        # tokenize=False: 获取拼接好的字符串
        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False 
        )
        prompts_text.append(text)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    print(f"开始推理 {len(data)} 条数据 ...")
    # --- 修改点：直接传入 prompts ---
    outputs = llm.generate(prompts=prompts_text, sampling_params=sampling_params)

    results = []
    for i, output in enumerate(outputs):
        completion = output.outputs[0]
        generated_text = completion.text
        token_ids = completion.token_ids
        token_count = len(token_ids)
        
        result_item = {
            "original_input": data[i],
            "model_response": generated_text,
            "response_token_length": token_count,
            "finish_reason": completion.finish_reason
        }
        results.append(result_item)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"推理完成！结果已保存至 {OUTPUT_FILE}")

    if results:
        token_counts = [item["response_token_length"] for item in results]
        avg_tokens = sum(token_counts) / len(token_counts)
        print("\n" + "="*35)
        print("📊 本次推理 Token 统计")
        print("="*35)
        print(f"总样本数: {len(token_counts)}")
        print(f"平均 Token 数: {avg_tokens:.2f}")
        print(f"最长 Token 数: {max(token_counts)}")
        print(f"最短 Token 数: {min(token_counts)}")
        print("="*35 + "\n")

if __name__ == "__main__":
    main()