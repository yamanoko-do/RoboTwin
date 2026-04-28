#!/bin/bash
#bash eval.sh beat_block_hammer demo_clean 0 0 /mnt/workspace/yama/RoboTwin/policy/DP/data/outputs/2026.04.27/23.36.15_beat_block_hammer_hhh/checkpoints/beat_block_hammer-demo_clean-50-0/600.ckpt
#                         task_name   config seed gpu_id  ckpt_path

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
seed=${3}
gpu_id=${4}
DEBUG=False

# == checkpoint path (required) ==
# Full path to the checkpoint file.
# Example: ckpt_path=/mnt/workspace/yama/RoboTwin/policy/DP/data/outputs/2026.04.27/23.36.15_beat_block_hammer_hhh/checkpoints/beat_block_hammer-demo_clean-50-0/600.ckpt
ckpt_path=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --ckpt_path "${ckpt_path}"