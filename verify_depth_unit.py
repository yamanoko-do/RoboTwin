"""
Verify the unit of depth data in RoboTwin collected datasets.

Logic:
  1. Load depth image + camera extrinsics from HDF5
  2. Unproject depth pixels back to 3D world coordinates
  3. Compare the reconstructed world Z (height) under two hypotheses:
     - Hypothesis A: depth is in meters  → world Z should be ~0.8m (table height)
     - Hypothesis B: depth is in millimeters → world Z should be ~0.8m (table height)
  4. Also compare the reconstructed camera-to-object distance against
     the known camera height (~1.35m), which is the upper bound of depth.
"""

import sys
import numpy as np
import h5py


def unproject_depth(depth, intrinsic_cv, extrinsic_cv):
    """Unproject depth map to world coordinates.

    Args:
        depth: (H, W) depth image (unit unknown)
        intrinsic_cv: (3, 3) OpenCV intrinsic matrix
        extrinsic_cv: (3, 4) OpenCV extrinsic matrix [R|t], world = R^T @ (cam - t)
    Returns:
        points_world: (N, 3) world coordinates for valid (non-zero) depth pixels
    """
    h, w = depth.shape
    fx = intrinsic_cv[0, 0]
    fy = intrinsic_cv[1, 1]
    cx = intrinsic_cv[0, 2]
    cy = intrinsic_cv[1, 2]

    v, u = np.where(depth > 0)
    d = depth[v, u]

    # Pixel -> camera coordinates (OpenCV convention: Z forward, X right, Y down)
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Camera -> world:  world = R^T @ (cam - t)
    R = extrinsic_cv[:3, :3]
    t = extrinsic_cv[:3, 3]
    pts_world = (R.T @ (pts_cam - t).T).T  # (N, 3)

    return pts_world


def verify(hdf5_path, episode=0, frame=0, camera="head_camera"):
    print(f"File:    {hdf5_path}")
    print(f"Episode: {episode}, Frame: {frame}, Camera: {camera}\n")

    with h5py.File(hdf5_path, "r") as f:
        depth_raw = f[f"observation/{camera}/depth"][frame]
        ext = f[f"observation/{camera}/extrinsic_cv"][frame]
        int_mat = f[f"observation/{camera}/intrinsic_cv"][frame]

    R = ext[:3, :3]
    t = ext[:3, 3]
    cam_pos_world = -R.T @ t
    cam_height = cam_pos_world[2]

    print(f"Camera world position: ({cam_pos_world[0]:.3f}, {cam_pos_world[1]:.3f}, {cam_pos_world[2]:.3f})")
    print(f"Camera height (Z):     {cam_height:.3f} m\n")

    valid_mask = depth_raw > 0
    valid_depth = depth_raw[valid_mask]

    print(f"Raw depth stats (non-zero pixels: {valid_mask.sum()} / {depth_raw.size}):")
    print(f"  min={valid_depth.min():.2f}  max={valid_depth.max():.2f}  "
          f"mean={valid_depth.mean():.2f}  median={np.median(valid_depth):.2f}\n")

    # --- Hypothesis A: depth in meters ---
    depth_m = depth_raw.copy().astype(np.float64)
    pts_m = unproject_depth(depth_m, int_mat, ext)
    print("=== Hypothesis A: depth unit = meters ===")
    print(f"  Depth range: {depth_m[valid_mask].min():.2f} ~ {depth_m[valid_mask].max():.2f} m")
    print(f"  Reconstructed world Z: min={pts_m[:, 2].min():.2f}, max={pts_m[:, 2].max():.2f}, "
          f"mean={pts_m[:, 2].mean():.2f} m")
    print(f"  Table height should be ~0.8 m → {'PASS' if 0.5 < pts_m[:, 2].mean() < 1.2 else 'FAIL (values way off)'}\n")

    # --- Hypothesis B: depth in millimeters ---
    depth_mm = depth_raw.copy().astype(np.float64) / 1000.0
    pts_mm = unproject_depth(depth_mm, int_mat, ext)
    print("=== Hypothesis B: depth unit = millimeters ===")
    print(f"  Depth range: {depth_mm[valid_mask].min():.4f} ~ {depth_mm[valid_mask].max():.4f} m")
    print(f"  Reconstructed world Z: min={pts_mm[:, 2].min():.4f}, max={pts_mm[:, 2].max():.4f}, "
          f"mean={pts_mm[:, 2].mean():.4f} m")
    print(f"  Table height should be ~0.8 m → {'PASS' if 0.5 < pts_mm[:, 2].mean() < 1.2 else 'FAIL (values way off)'}\n")

    # --- Sanity check: depth vs camera height ---
    print("=== Sanity check: object distance vs camera height ===")
    print(f"  Camera height = {cam_height:.3f} m")
    print(f"  If depth is meters:   object distance = {depth_m[valid_mask].mean():.2f} m  "
          f"→ {'reasonable' if depth_m[valid_mask].mean() < cam_height else 'ABOVE camera, impossible'}")
    print(f"  If depth is mm:       object distance = {depth_mm[valid_mask].mean():.4f} m  "
          f"→ {'reasonable' if depth_mm[valid_mask].mean() < cam_height else 'ABOVE camera, impossible'}\n")

    # --- Verdict ---
    mean_mm = pts_mm[:, 2].mean()
    mean_m = pts_m[:, 2].mean()
    dist_mm = depth_mm[valid_mask].mean()
    dist_m = depth_m[valid_mask].mean()

    score_mm = abs(mean_mm - 0.8)
    score_m = abs(mean_m - 0.8) if dist_m < cam_height else float("inf")

    verdict = "millimeters" if score_mm < score_m else "meters"
    print(f"=== VERDICT: depth unit is likely '{verdict}' ===")


if __name__ == "__main__":
    hdf5_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/workspace/yama/robotwin_data/move_playingcard_away/demo_clean/data/episode0.hdf5"
    episode = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    camera = sys.argv[4] if len(sys.argv) > 4 else "head_camera"

    verify(hdf5_path, episode, frame, camera)
