CUDA_VISIBLE_DEVICES=0 vllm serve casperhansen/deepseek-r1-distill-llama-70b-awq \
  --quantization awq \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --download-dir ./llm