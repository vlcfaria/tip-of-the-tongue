CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 vllm serve casperhansen/deepseek-r1-distill-llama-70b-awq \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --download-dir ./llm