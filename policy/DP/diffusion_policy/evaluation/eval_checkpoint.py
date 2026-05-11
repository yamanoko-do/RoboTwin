"""
Standalone evaluation script — loads a checkpoint and runs policy evaluation.
Launched as a subprocess by the training loop so that SAPIEN simulation
does not interfere with NCCL distributed training.

Can run in parallel across multiple GPUs (each GPU is a separate process).
The shell script aggregates results and logs to SwanLab after all processes finish.

Usage:
    # Single-GPU mode:
    python eval_checkpoint.py \
        --checkpoint /path/to/epoch.ckpt \
        --eval_task_name move_playingcard_away \
        --eval_task_config demo_clean \
        --head_camera_type D435 \
        --num_episodes 8 \
        --instruction_type unseen \
        --output_dir /path/to/output \
        --epoch 0

    # Multi-GPU mode (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_checkpoint.py --rank 0 --world_size 8 ...
    CUDA_VISIBLE_DEVICES=1 python eval_checkpoint.py --rank 1 --world_size 8 ...
"""
import argparse
import os
import sys
import pathlib
import json

# Ensure project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DP_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _DP_ROOT not in sys.path:
    sys.path.insert(0, _DP_ROOT)
_ROBOTWIN_ROOT = os.path.abspath(os.path.join(_DP_ROOT, "..", "..", ".."))
if _ROBOTWIN_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTWIN_ROOT)

import torch
import dill
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.evaluation.in_training_evaluator import find_successful_seeds, run_eval_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_task_name", required=True)
    parser.add_argument("--eval_task_config", required=True)
    parser.add_argument("--head_camera_type", required=True)
    parser.add_argument("--num_episodes", type=int, default=8)
    parser.add_argument("--instruction_type", default="unseen")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--experiment_name", default="")
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES is set by the parent process via env,
    # so torch only sees the designated GPU. Use cuda:0 (the only visible device).
    device = "cuda:0"

    # Load checkpoint
    print(f"\033[93m[Eval rank {args.rank}] Loading checkpoint from {args.checkpoint} on {device}...\033[0m")
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # Reconstruct workspace and load weights
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload)

    # Get policy (prefer EMA)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.to(device)

    # Find successful seeds (only rank 0 searches, then broadcasts)
    eval_seeds = None
    if args.rank == 0:
        eval_seeds = find_successful_seeds(
            task_name=args.eval_task_name,
            task_config=args.eval_task_config,
            seed=cfg.training.seed,
            head_camera_type=args.head_camera_type,
            num_episodes=args.num_episodes,
        )

    # Distribute seeds across ranks via file
    seeds_file = os.path.join(args.output_dir, f"eval_epoch{args.epoch}_seeds.json")
    if args.rank == 0 and eval_seeds is not None:
        with open(seeds_file, "w") as f:
            json.dump(eval_seeds, f)

    # Other ranks wait for the seeds file
    if args.world_size > 1 and args.rank != 0:
        import time
        for _ in range(300):  # wait up to 5 min
            if os.path.exists(seeds_file):
                break
            time.sleep(1)
        else:
            print(f"\033[91m[Eval rank {args.rank}] Timed out waiting for seeds file\033[0m")
            return
        with open(seeds_file, "r") as f:
            eval_seeds = json.load(f)

    # Episodes for this rank
    my_episodes = args.num_episodes // args.world_size
    remainder = args.num_episodes % args.world_size
    if args.rank < remainder:
        my_episodes += 1

    # Run evaluation
    # Seeds already distributed across ranks via slicing above.
    # Pass rank=0, world_size=1 so run_eval_episodes uses them as-is.
    my_seeds = eval_seeds[args.rank::args.world_size] if eval_seeds else None

    success_rate, num_success, num_total, video_path = run_eval_episodes(
        policy=policy,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        task_name=args.eval_task_name,
        task_config=args.eval_task_config,
        seed=cfg.training.seed,
        head_camera_type=args.head_camera_type,
        num_episodes=my_episodes,
        instruction_type=args.instruction_type,
        rank=0,
        world_size=1,
        pre_collected_seeds=my_seeds,
        epoch=args.epoch,
        gpu_rank=args.rank,
        experiment_name=args.experiment_name,
    )

    print(f"\033[96m[Eval rank {args.rank}] Epoch {args.epoch}: {num_success}/{num_total} = {success_rate:.1%}\033[0m")

    # Each rank writes its own partial result JSON
    rank_result_path = os.path.join(args.output_dir, f"eval_epoch{args.epoch}_rank{args.rank}.json")
    with open(rank_result_path, "w") as f:
        json.dump({
            "epoch": args.epoch,
            "rank": args.rank,
            "success_count": num_success,
            "total_episodes": num_total,
            "video_path": video_path,
        }, f)
    print(f"\033[96m[Eval rank {args.rank}] Result saved to {rank_result_path}\033[0m")

    # Cleanup seeds file (rank 0 only)
    if args.rank == 0 and os.path.exists(seeds_file):
        os.remove(seeds_file)


if __name__ == "__main__":
    main()
