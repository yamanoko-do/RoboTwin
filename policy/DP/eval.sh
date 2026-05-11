#!/bin/bash
# Usage:
#   bash eval.sh --task_name beat_block_hammer --task_config demo_clean --seed 0 --gpu_id 0 --ckpt_path /path/to/ckpt [--test_num 20]
#
# Legacy positional args also supported:
#   bash eval.sh task_name config seed gpu_id ckpt_path [--test_num 20]

policy_name=DP
DEBUG=False

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name)  task_name="$2";  shift 2 ;;
        --task_config) task_config="$2"; shift 2 ;;
        --seed)       seed="$2";       shift 2 ;;
        --gpu_id)     gpu_id="$2";     shift 2 ;;
        --ckpt_path)  ckpt_path="$2";  shift 2 ;;
        --test_num)   test_num="--test_num $2"; shift 2 ;;
        *)            extra_args="$extra_args $1"; shift ;;
    esac
done

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --ckpt_path "${ckpt_path}" \
    ${test_num} ${extra_args}
