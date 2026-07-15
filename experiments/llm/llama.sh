CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --served-model-name tip-of-tongue-rewriter \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --download-dir ./llama