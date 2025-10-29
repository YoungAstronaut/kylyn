# 在目标机上，直接限制到卡 3
export CUDA_VISIBLE_DEVICES=3
export VLLM_API_KEY="secret-embed-key"

vllm serve ../llm_models/Qwen/Qwen3-Embedding-4B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8005 \
  --served-model-name qwen3-embed-4b
