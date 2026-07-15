CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model ./models/tot_sft_merged_model \
    --served-model-name tip-of-tongue-rewriter-base \
    --quantization fp8 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules simpo-aligned-rewriter=./models/simpo-query-rewriter-final