#!/bin/bash

# 4 gpus
# python rm_bench_scripts/run_rm.py \
#     --model TIGER-Lab/AceCodeRM-7B \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 16 \
#     --trust_remote_code 

# python rm_bench_scripts/run_rm.py \
#     --model TIGER-Lab/AceCodeRM-32B \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 16 \
#     --trust_remote_code 

# python rm_bench_scripts/run_rm.py \
#     --model Skywork/Skywork-Reward-Gemma-2-27B-v0.2 \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 1 \
#     --trust_remote_code

# python rm_bench_scripts/run_rm.py \
#     --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 16 \
#     --trust_remote_code 

# python rm_bench_scripts/run_rm.py \
#     --model internlm/internlm2-20b-reward \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 1 \
#     --trust_remote_code 

python rm_bench_scripts/run_rm.py \
    --model internlm/internlm2-7b-reward \
    --datapath rm_bench_data/total_dataset.json \
    --batch_size 1 \
    --trust_remote_code 

    