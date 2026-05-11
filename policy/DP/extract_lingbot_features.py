"""
Pre-extract LingBotDepthEncoder features from zarr data.

One run extracts ALL feature arrays (3 cameras x 2 modes x 2 feature types = 12 total):
  CLS tokens (existing):
    - {cam}_lingbot_cls_rgb   : (N, 1024) — CLS token from RGB-only mode
    - {cam}_lingbot_cls_rgbd  : (N, 1024) — CLS token from RGB+Depth mode

  Spatial features (new):
    - {cam}_lingbot_spatial_rgb   : (N, 1024, pool_h, pool_w) — pooled spatial features, RGB-only
    - {cam}_lingbot_spatial_rgbd  : (N, 1024, pool_h, pool_w) — pooled spatial features, RGB+Depth

The spatial features are raw (no CLS injection) — CLS injection happens at training time
so that pool(features + cls) == pool(features) + cls[...,None,None] (linearity of avg pool).

Idempotent — skips keys that already exist.

Usage:
    python extract_lingbot_features.py \
        --zarr_path ./data/move_playingcard_away-demo_clean-50.zarr

After extraction, the training configs select the appropriate keys:
  - CLS-only (legacy):   uses *_cls_rgbd
  - Spatial (new):        uses both *_cls_rgbd and *_spatial_rgbd
"""
import argparse
import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import zarr
from tqdm import tqdm

from diffusion_policy.model.vision.lingbot_depth_encoder import LingBotDepthEncoder


class _ClsExtractor(nn.Module):
    """Thin wrapper so DataParallel can parallelize CLS extraction."""
    def __init__(self, encoder: LingBotDepthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, rgb, depth=None):
        return self.encoder.extract_cls_token(rgb, depth=depth)


class _SpatialExtractor(nn.Module):
    """Thin wrapper so DataParallel can parallelize spatial feature extraction."""
    def __init__(self, encoder: LingBotDepthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, rgb, depth=None):
        return self.encoder.extract_spatial_features(rgb, depth=depth)


# Maps from short cam name to the raw zarr keys
_CAM_KEYS = {
    "head_cam": ("head_camera", "head_camera_depth"),
    "left_cam": ("left_camera", "left_camera_depth"),
    "right_cam": ("right_camera", "right_camera_depth"),
}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs for DataParallel")
    parser.add_argument("--pretrained_path", default="/mnt/data/cpfs/yama/lingbotDepth/model.pt")
    parser.add_argument("--pool_h", type=int, default=6,
                        help="Spatial feature pool height (default 6 for 4:3 aspect ratio)")
    parser.add_argument("--pool_w", type=int, default=8,
                        help="Spatial feature pool width (default 8 for 4:3 aspect ratio)")
    parser.add_argument("--skip_cls", action="store_true",
                        help="Skip CLS token extraction (only extract spatial features)")
    parser.add_argument("--skip_spatial", action="store_true",
                        help="Skip spatial feature extraction (only extract CLS tokens)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")

    # Load encoder with spatial features enabled
    encoder = LingBotDepthEncoder(
        pretrained_path=args.pretrained_path,
        feature_dim=512,
        freeze_backbone=True,
        resolution_level=5,
        use_fp16=True,
        use_spatial_features=True,
        pool_h=args.pool_h,
        pool_w=args.pool_w,
    ).to(device).eval()

    cls_dim = encoder.mdm_model.encoder.dim_features  # e.g. 1024 for ViT-L

    # Multi-GPU: wrap extraction helpers with DataParallel
    # We must call forward() for DataParallel to split data across GPUs;
    # calling encoder.module.method() directly runs on a single GPU only.
    cls_extractor = _ClsExtractor(encoder)
    spatial_extractor = _SpatialExtractor(encoder)
    if len(gpu_ids) > 1:
        cls_extractor = nn.DataParallel(cls_extractor, device_ids=gpu_ids)
        spatial_extractor = nn.DataParallel(spatial_extractor, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using single GPU: {gpu_ids[0]}")

    # Open zarr
    root = zarr.open(args.zarr_path, mode="r+")
    data = root["data"]

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for cam_name, (rgb_zarr_key, depth_zarr_key) in _CAM_KEYS.items():
        cam_data = data[rgb_zarr_key]  # (N, 3, H, W) uint8
        depth_data = data[depth_zarr_key]  # (N, 1, H, W) float32
        n_frames = cam_data.shape[0]

        for mode in ["rgb", "rgbd"]:
            # === CLS token extraction ===
            cls_key = f"{cam_name}_lingbot_cls_{mode}"
            if not args.skip_cls:
                if cls_key in data:
                    print(f"[skip] {cls_key} already exists in zarr")
                else:
                    print(f"Extracting {cls_key} (n={n_frames}, mode={mode}) ...")
                    cls_tokens = np.empty((n_frames, cls_dim), dtype=np.float32)

                    for start in tqdm(range(0, n_frames, args.batch_size),
                                      desc=cls_key):
                        end = min(start + args.batch_size, n_frames)

                        rgb = torch.from_numpy(np.array(cam_data[start:end])).float() / 255.0
                        rgb = rgb.to(device)

                        depth = None
                        if mode == "rgbd":
                            depth = torch.from_numpy(
                                np.array(depth_data[start:end]).astype(np.float32)
                            ) / args.depth_scale
                            depth = depth.to(device)

                        cls = cls_extractor(rgb, depth)
                        cls_tokens[start:end] = cls.cpu().numpy()

                    data.create_dataset(
                        cls_key,
                        data=cls_tokens,
                        chunks=(100, cls_dim),
                        dtype="float32",
                        overwrite=True,
                        compressor=compressor,
                    )
                    print(f"[done] {cls_key} shape={cls_tokens.shape}")

            # === Spatial feature extraction ===
            spatial_key = f"{cam_name}_lingbot_spatial_{mode}"
            if not args.skip_spatial:
                if spatial_key in data:
                    print(f"[skip] {spatial_key} already exists in zarr")
                else:
                    print(f"Extracting {spatial_key} (n={n_frames}, mode={mode}, "
                          f"pool={args.pool_h}x{args.pool_w}) ...")
                    spatial_feats = np.empty(
                        (n_frames, cls_dim, args.pool_h, args.pool_w),
                        dtype=np.float32,
                    )

                    for start in tqdm(range(0, n_frames, args.batch_size),
                                      desc=spatial_key):
                        end = min(start + args.batch_size, n_frames)

                        rgb = torch.from_numpy(np.array(cam_data[start:end])).float() / 255.0
                        rgb = rgb.to(device)

                        depth = None
                        if mode == "rgbd":
                            depth = torch.from_numpy(
                                np.array(depth_data[start:end]).astype(np.float32)
                            ) / args.depth_scale
                            depth = depth.to(device)

                        feats_pooled, _ = spatial_extractor(rgb, depth)
                        spatial_feats[start:end] = feats_pooled.cpu().numpy()

                    data.create_dataset(
                        spatial_key,
                        data=spatial_feats,
                        chunks=(100, cls_dim, args.pool_h, args.pool_w),
                        dtype="float32",
                        overwrite=True,
                        compressor=compressor,
                    )
                    print(f"[done] {spatial_key} shape={spatial_feats.shape}")

    print("All features extracted.")


if __name__ == "__main__":
    main()
