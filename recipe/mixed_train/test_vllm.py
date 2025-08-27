import time
import numpy as np
from vllm import LLM, SamplingParams

# ===== 1. 多卡并行配置 =====
model_name = "../llm_models/Qwen/Qwen2.5-7B-Instruct"  # 支持替换为 Llama-3-70B, DeepSeek-R1 等
tensor_parallel_size = 4                  # 使用4卡张量并行
max_model_len = 8192                      # 支持长上下文

# ===== 2. 生成512个真实场景Prompt =====
prompts = [
    "请解释量子纠缠的原理及其在量子通信中的应用。" + "要求：分三点说明，每点不少于50字。" * 5
    for _ in range(512)  # 生成512条长文本Prompt
]
print(f"✅ 已生成 {len(prompts)} 条测试Prompt，平均长度：{len(prompts[0])}字符")

# ===== 3. 初始化模型（启用4卡并行）=====
llm = LLM(
    model=model_name,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
    gpu_memory_utilization=0.80,         # 显存利用率调优
    trust_remote_code=True,
    enforce_eager=True,                    # 避免内核编译开销
)

# ===== 4. 采样参数（模拟真实生成）=====
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,                       # 每条输出限制256 token
    skip_special_tokens=True
)

# ===== 5. 执行推理并测量耗时 =====
start_time = time.time()
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
end_time = time.time()
total_time = end_time - start_time

# ===== 6. 性能分析 =====
avg_time_per_prompt = total_time / len(prompts)
throughput_tokens = sum(len(out.outputs[0].token_ids) for out in outputs) / total_time

# 首Token延迟（TTFT）统计
# first_token_latencies = []
# for out in outputs:
#     if out.outputs:
#         first_token_time = out.outputs[0].timestamp - start_time
#         first_token_latencies.append(first_token_time)

# ===== 7. 结果输出 =====
print(f"\n🔁 总Prompt数量: {len(prompts)}")
print(f"⏱️ 总耗时: {total_time:.2f}秒")
print(f"🚀 吞吐量: {len(prompts)/total_time:.2f} prompt/秒 | {throughput_tokens:.2f} token/秒")
print(f"⏳ 平均单Prompt延迟: {avg_time_per_prompt:.3f}秒")
# print(f"⚡ 首Token延迟 (TTFT): P50={np.percentile(first_token_latencies, 50):.3f}s | P90={np.percentile(first_token_latencies, 90):.3f}s")