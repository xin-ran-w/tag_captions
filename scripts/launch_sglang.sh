gpus="0,1,2,3,4,5,6,7,8"
n_gpu=8

CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server \
    --model-path /home/diaomuxi/model_zoo/Qwen2.5-VL-7B-Instruct \
    --chat-template=qwen2-vl \
    --dp $n_gpu

# CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server \
#     --model-path /home/diaomuxi/model_zoo/gemma-3-4b-it \
#     --chat-template=gemma-it \
#     --context-len 32000 \
#     --dp $n_gpu

# CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server \
#     --model-path /home/diaomuxi/model_zoo/deepseek-vl2 \
#     --chat-template=deepseek-vl2 \
#     --context-len 4096 \
#     --dp $n_gpu
    