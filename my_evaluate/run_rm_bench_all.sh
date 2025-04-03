#!/bin/bash

# python rm_bench_scripts/run_rm.py \
#     --model TIGER-Lab/AceCodeRM-7B \
#     --datapath rm_bench_data/total_dataset.json \
#     --batch_size 16 \
#     --trust_remote_code 

python rm_bench_scripts/run_rm.py \
    --model TIGER-Lab/AceCodeRM-32B \
    --datapath rm_bench_data/total_dataset.json \
    --batch_size 1 \
    --trust_remote_code 