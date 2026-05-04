import os
import sys
import subprocess
import shutil
import tempfile
import importlib

import numpy as np
import yaml

from diffusion_policy.env_runner.dp_runner import DPRunner


# Ensure RoboTwin root and description/utils are on sys.path
_CURRENT_FILE = os.path.abspath(__file__)
# This file is at: RoboTwin/policy/DP/diffusion_policy/evaluation/in_training_evaluator.py
_ROBOTWIN_ROOT = os.path.abspath(os.path.join(_CURRENT_FILE, "..", "..", "..", "..", ".."))
if _ROBOTWIN_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTWIN_ROOT)
_DESCRIPTION_UTILS = os.path.join(_ROBOTWIN_ROOT, "description", "utils")
if _DESCRIPTION_UTILS not in sys.path:
    sys.path.insert(0, _DESCRIPTION_UTILS)


def _class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


class InTrainingDPModel:
    """Drop-in replacement for DP that uses an already-loaded policy."""

    def __init__(self, policy, n_obs_steps, n_action_steps):
        self.policy = policy
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)


def _prepare_args(task_name, task_config, head_camera_type):
    """Build the args dict that the task environment expects, mirroring eval_policy.py main()."""
    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError  # noqa: F401 — ensure importable

    with open(os.path.join(_ROBOTWIN_ROOT, "task_config", f"{task_config}.yml"), "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = "DP"

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "camera" not in args:
        args["camera"] = {}
    if head_camera_type:
        args["camera"]["head_camera_type"] = head_camera_type
    cam_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[cam_type]["h"]
    args["head_camera_w"] = _camera_config[cam_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    robot_config_file = os.path.join(args["left_robot_file"], "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        left_embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    robot_config_file = os.path.join(args["right_robot_file"], "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        right_embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = left_embodiment_config
    args["right_embodiment_config"] = right_embodiment_config
    args["left_arm_dim"] = len(left_embodiment_config["arm_joints_name"][0])
    args["right_arm_dim"] = len(right_embodiment_config["arm_joints_name"][1])

    video_size = f"{_camera_config[cam_type]['w']}x{_camera_config[cam_type]['h']}"
    return args, video_size


def find_successful_seeds(task_name, task_config, seed, head_camera_type,
                          num_episodes=8, max_attempts=500):
    """
    Run expert demonstrations to find seeds where the expert succeeds.
    This is called once before training starts, on rank 0 only.

    Returns:
        list of int: seeds where expert check passed, length == num_episodes
    """
    from envs.utils.create_actor import UnStableError

    args, _ = _prepare_args(task_name, task_config, head_camera_type)
    args["render_freq"] = 0
    args["eval_mode"] = True
    args["gpu_id"] = 0  # rank 0 only

    TASK_ENV = _class_decorator(task_name)

    st_seed = 100000 * (1 + seed)
    successful_seeds = []
    now_seed = st_seed
    now_id = 0

    print(f"\033[93m[SeedSearch] Searching for {num_episodes} successful seeds (max {max_attempts} attempts)...\033[0m")

    while len(successful_seeds) < num_episodes and now_seed - st_seed < max_attempts:
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except (UnStableError, Exception):
            TASK_ENV.close_env()
            now_seed += 1
            continue

        if TASK_ENV.plan_success and TASK_ENV.check_success():
            successful_seeds.append(now_seed)
            print(f"\033[92m[SeedSearch] Found seed {now_seed} ({len(successful_seeds)}/{num_episodes})\033[0m")
        now_seed += 1
        now_id += 1

    if len(successful_seeds) < num_episodes:
        print(f"\033[91m[SeedSearch] Warning: only found {len(successful_seeds)} successful seeds "
              f"after {max_attempts} attempts\033[0m")

    print(f"\033[96m[SeedSearch] Done. Seeds: {successful_seeds}\033[0m")
    return successful_seeds


def run_eval_episodes(policy, n_obs_steps, n_action_steps,
                      task_name, task_config, seed, head_camera_type,
                      num_episodes=10, instruction_type="unseen",
                      rank=0, world_size=1,
                      pre_collected_seeds=None):
    """
    Run policy evaluation during training.

    Args:
        pre_collected_seeds: list of seeds where expert check already passed.
            If provided, skip expert_check and use these seeds directly,
            distributed across ranks. This avoids the NCCL timeout caused by
            uneven expert_check retry times across ranks.

    Returns:
        (success_rate, num_success, num_total, selected_video_path)
    """
    from envs.utils.create_actor import UnStableError
    from sapien.render import clear_cache as sapien_clear_cache

    # import DP deploy functions
    dp_deploy = importlib.import_module("policy.DP.deploy_policy")
    eval_func = dp_deploy.eval
    reset_func = dp_deploy.reset_model

    # import instruction generator
    from generate_episode_instructions import generate_episode_descriptions

    args, video_size = _prepare_args(task_name, task_config, head_camera_type)

    # Set gpu_id for SAPIEN renderer device isolation
    gpu_id = int(os.environ.get("LOCAL_RANK", rank))
    args["gpu_id"] = gpu_id

    # Setup video recording
    tmp_video_dir = tempfile.mkdtemp(prefix="tmp_eval_video_")
    args["eval_video_log"] = True
    args["eval_video_save_dir"] = tmp_video_dir
    args["render_freq"] = 0
    args["eval_mode"] = True

    model = InTrainingDPModel(policy, n_obs_steps, n_action_steps)
    TASK_ENV = _class_decorator(task_name)

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    clear_cache_freq = args["clear_cache_freq"]
    episode_records = []  # (video_path, success)

    if pre_collected_seeds is not None:
        # Use pre-collected seeds: distribute across ranks
        my_seeds = pre_collected_seeds[rank::world_size]
        print(f"\033[93m[Eval] Rank {rank} running {len(my_seeds)} episodes with pre-collected seeds {my_seeds}\033[0m")
        for ep_idx, eval_seed in enumerate(my_seeds):
            # Setup episode for policy evaluation
            TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=eval_seed, is_test=True, **args)
            # Get episode info for instruction generation
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
            # Re-setup for policy eval
            TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=eval_seed, is_test=True, **args)
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(task_name, episode_info_list, num_episodes)
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)

            # Start ffmpeg for video recording
            if TASK_ENV.eval_video_path is not None:
                ep_video_path = os.path.join(TASK_ENV.eval_video_path, f"episode{TASK_ENV.test_num}.mp4")
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", video_size, "-framerate", "10",
                        "-i", "-",
                        "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                        "-crf", "23", ep_video_path,
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            # Run policy
            succ = False
            reset_func(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()

            if succ:
                TASK_ENV.suc += 1
                print(f"\033[92m[Eval] Success!\033[0m")
            else:
                print(f"\033[91m[Eval] Fail!\033[0m")

            # Record episode video
            ep_video_path = os.path.join(tmp_video_dir, f"episode{TASK_ENV.test_num}.mp4")
            if os.path.exists(ep_video_path):
                episode_records.append((ep_video_path, succ))

            TASK_ENV.close_env(clear_cache=((ep_idx + 1) % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            TASK_ENV.test_num += 1
            print(
                f"[Eval] {task_name} | {task_config}\n"
                f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "
                f"{round(TASK_ENV.suc / max(TASK_ENV.test_num, 1) * 100, 1)}%"
            )
    else:
        # Fallback: original expert_check logic (for single-GPU or non-distributed use)
        st_seed = 100000 * (1 + seed)
        local_num_episodes = (num_episodes + world_size - 1) // world_size
        st_seed = st_seed + rank * 1000
        now_seed = st_seed
        now_id = 0
        succ_seed = 0
        expert_check = True

        while succ_seed < local_num_episodes:
            render_freq = args["render_freq"]
            args["render_freq"] = 0

            if expert_check:
                try:
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                    episode_info = TASK_ENV.play_once()
                    TASK_ENV.close_env()
                except UnStableError:
                    TASK_ENV.close_env()
                    now_seed += 1
                    args["render_freq"] = render_freq
                    continue
                except Exception:
                    TASK_ENV.close_env()
                    now_seed += 1
                    args["render_freq"] = render_freq
                    continue

            if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
                succ_seed += 1
            else:
                now_seed += 1
                args["render_freq"] = render_freq
                continue

            args["render_freq"] = render_freq

            # Setup episode for policy evaluation
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(task_name, episode_info_list, num_episodes)
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)

            # Start ffmpeg for video recording
            if TASK_ENV.eval_video_path is not None:
                ep_video_path = os.path.join(TASK_ENV.eval_video_path, f"episode{TASK_ENV.test_num}.mp4")
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", video_size, "-framerate", "10",
                        "-i", "-",
                        "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                        "-crf", "23", ep_video_path,
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            # Run policy
            succ = False
            reset_func(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()

            if succ:
                TASK_ENV.suc += 1
                print(f"\033[92m[Eval] Success!\033[0m")
            else:
                print(f"\033[91m[Eval] Fail!\033[0m")

            # Record episode video
            ep_video_path = os.path.join(tmp_video_dir, f"episode{TASK_ENV.test_num}.mp4")
            if os.path.exists(ep_video_path):
                episode_records.append((ep_video_path, succ))

            now_id += 1
            TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            TASK_ENV.test_num += 1
            print(
                f"[Eval] {task_name} | {task_config}\n"
                f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "
                f"{round(TASK_ENV.suc / max(TASK_ENV.test_num, 1) * 100, 1)}%"
            )
            now_seed += 1

    # Select video: first successful, or first episode
    selected_video = None
    for ep_path, ep_succ in episode_records:
        if ep_succ:
            selected_video = ep_path
            break
    if selected_video is None and episode_records:
        selected_video = episode_records[0][0]

    # Copy selected video to a persistent location before cleaning tmp
    if selected_video is not None:
        persistent_dir = os.path.join(_ROBOTWIN_ROOT, "policy", "DP", "data", "eval_videos")
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_path = os.path.join(persistent_dir, f"epoch_eval_{task_name}.mp4")
        shutil.copy2(selected_video, persistent_path)
        selected_video = persistent_path

    # Cleanup
    sapien_clear_cache()
    shutil.rmtree(tmp_video_dir, ignore_errors=True)

    success_rate = TASK_ENV.suc / max(TASK_ENV.test_num, 1)
    return success_rate, TASK_ENV.suc, TASK_ENV.test_num, selected_video
