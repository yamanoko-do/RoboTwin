"""
Aggregate per-rank eval results and log to SwanLab.

Called by train_eval_loop.sh after all eval_checkpoint.py processes finish.
Since this is a fresh Python process each time, code changes take effect immediately
without restarting the training loop.

Usage:
    python aggregate_and_log.py \
        --output_dir /path/to/output \
        --epoch 1000 \
        --ckpt_path /path/to/epoch.ckpt \
        --swanlab_id_file /path/to/.swanlab_run_id \
        --eval_videos_dir /path/to/eval_videos
"""
import argparse
import json
import os
import glob
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--swanlab_id_file", required=True)
    parser.add_argument("--eval_videos_dir", required=True)
    parser.add_argument("--experiment_name", default="")
    args = parser.parse_args()

    epoch = args.epoch
    output_dir = args.output_dir
    swanlab_id_file = args.swanlab_id_file

    # Collect per-rank results
    total_success = 0
    total_episodes = 0
    rank_files = sorted(glob.glob(os.path.join(output_dir, f"eval_epoch{epoch}_rank*.json")))
    for rf in rank_files:
        with open(rf) as f:
            d = json.load(f)
        total_success += d["success_count"]
        total_episodes += d["total_episodes"]
        os.remove(rf)

    success_rate = total_success / max(total_episodes, 1)
    print(f"\033[96m[Eval] Epoch {epoch}: {total_success}/{total_episodes} = {success_rate:.1%}\033[0m")

    # Save combined result
    result_path = os.path.join(output_dir, f"eval_epoch{epoch}.json")
    with open(result_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "success_rate": success_rate,
            "success_count": total_success,
            "total_episodes": total_episodes,
        }, f)

    # Find video: prefer success, fallback to any
    exp_dir = args.experiment_name if args.experiment_name else ""
    video_base = os.path.join(args.eval_videos_dir, exp_dir) if exp_dir else args.eval_videos_dir
    video_dir = os.path.join(video_base, f"eval_epoch{epoch}")
    success_videos = sorted(glob.glob(os.path.join(video_dir, "*_success.mp4"))) if os.path.isdir(video_dir) else []
    all_videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4"))) if os.path.isdir(video_dir) else []
    best_video = success_videos[0] if success_videos else (all_videos[0] if all_videos else None)

    print(f"[Aggregate] video_dir={video_dir}, success={len(success_videos)}, total={len(all_videos)}, best={best_video}")

    # Log to SwanLab
    try:
        import swanlab
        import torch
        import dill
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        payload = torch.load(open(args.ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]

        init_kwargs = dict(
            workspace="limxooo",
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.logging.get("mode", "disabled"),
        )
        for k, v in cfg.logging.items():
            if k != "mode":
                init_kwargs[k] = v

        # Resume existing run
        if os.path.exists(swanlab_id_file):
            with open(swanlab_id_file) as f:
                sid = f.read().strip()
            if sid:
                init_kwargs["id"] = sid
                init_kwargs["resume"] = "must"

        swanlab.init(**init_kwargs)
        swanlab.log({
            "eval/success_rate": success_rate,
            "eval/success_count": total_success,
            "eval/total_episodes": total_episodes,
        }, step=epoch)

        # Upload video
        if best_video and os.path.exists(best_video):
            gif_path = best_video.replace(".mp4", ".gif")
            os.system(f"ffmpeg -y -loglevel error -i {best_video} -r 10 {gif_path}")
            if os.path.exists(gif_path):
                caption = f"epoch{epoch} ({'success' if total_success > 0 else 'fail'})"
                swanlab.log({"eval/video": swanlab.Video(gif_path, caption=caption)}, step=epoch)
                os.remove(gif_path)

        swanlab.finish()
    except Exception as e:
        print(f"\033[91m[Eval] SwanLab logging failed: {e}\033[0m")


if __name__ == "__main__":
    main()
