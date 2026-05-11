from typing import Dict, Optional
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_depth_identity_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        depth_scale: float = 1.0,
        # Pre-extracted LingBotDepth feature keys: maps rgbd rgb_key -> zarr feat key
        # e.g. {"head_cam": "head_cam_lingbot_feat"}
        preextracted_rgbd_feat_keys: Optional[Dict[str, str]] = None,
        # Pre-extracted LingBotDepth spatial feature keys: maps rgbd rgb_key -> zarr spatial key
        # e.g. {"head_cam": "head_cam_lingbot_spatial_rgbd"}
        preextracted_rgbd_spatial_keys: Optional[Dict[str, str]] = None,
    ):

        super().__init__()
        self.preextracted_rgbd_feat_keys = preextracted_rgbd_feat_keys or {}
        self.preextracted_rgbd_spatial_keys = preextracted_rgbd_spatial_keys or {}

        # Build the list of zarr keys to load
        zarr_keys = ["head_camera", "left_camera", "right_camera",
                     "head_camera_depth", "left_camera_depth", "right_camera_depth",
                     "state", "action"]
        # Add pre-extracted feature keys
        for feat_key in self.preextracted_rgbd_feat_keys.values():
            if feat_key not in zarr_keys:
                zarr_keys.append(feat_key)
        for spatial_key in self.preextracted_rgbd_spatial_keys.values():
            if spatial_key not in zarr_keys:
                zarr_keys.append(spatial_key)

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=zarr_keys,
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

        self.depth_scale = depth_scale

    def _resize_buffers(self, new_batch_size):
        sequence_length = self.sampler.sequence_length
        self.batch_size = new_batch_size
        self.buffers = {
            k: np.zeros((new_batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        if self.depth_scale > 1.0:
            # Depth is in meters after scaling — use identity normalizer
            # so LingBotDepth receives raw meter values
            normalizer["head_depth"] = get_depth_identity_normalizer()
            normalizer["left_depth"] = get_depth_identity_normalizer()
            normalizer["right_depth"] = get_depth_identity_normalizer()
        else:
            normalizer["head_depth"] = get_image_range_normalizer()
            normalizer["left_depth"] = get_image_range_normalizer()
            normalizer["right_depth"] = get_image_range_normalizer()
        # Pre-extracted features use identity normalizer (pass through)
        for feat_key in self.preextracted_rgbd_feat_keys.values():
            normalizer[feat_key] = get_depth_identity_normalizer()
        for spatial_key in self.preextracted_rgbd_spatial_keys.values():
            normalizer[spatial_key] = get_depth_identity_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        left_cam = np.moveaxis(sample["left_camera"], -1, 1) / 255
        right_cam = np.moveaxis(sample["right_camera"], -1, 1) / 255
        head_depth = sample["head_camera_depth"].astype(np.float32) / self.depth_scale
        left_depth = sample["left_camera_depth"].astype(np.float32) / self.depth_scale
        right_depth = sample["right_camera_depth"].astype(np.float32) / self.depth_scale

        obs = {
            "head_cam": head_cam,
            "left_cam": left_cam,
            "right_cam": right_cam,
            "head_depth": head_depth,
            "left_depth": left_depth,
            "right_depth": right_depth,
            "agent_pos": agent_pos,
        }
        for feat_key in self.preextracted_rgbd_feat_keys.values():
            obs[feat_key] = sample[feat_key].astype(np.float32)
        for spatial_key in self.preextracted_rgbd_spatial_keys.values():
            obs[spatial_key] = sample[spatial_key].astype(np.float32)

        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        left_cam = samples["left_camera"].to(device, non_blocking=True) / 255.0
        right_cam = samples["right_camera"].to(device, non_blocking=True) / 255.0
        head_depth = samples["head_camera_depth"].to(device, non_blocking=True) / self.depth_scale
        left_depth = samples["left_camera_depth"].to(device, non_blocking=True) / self.depth_scale
        right_depth = samples["right_camera_depth"].to(device, non_blocking=True) / self.depth_scale
        action = samples["action"].to(device, non_blocking=True)

        obs = {
            "head_cam": head_cam,
            "left_cam": left_cam,
            "right_cam": right_cam,
            "head_depth": head_depth,
            "left_depth": left_depth,
            "right_depth": right_depth,
            "agent_pos": agent_pos,
        }
        # Add pre-extracted LingBotDepth features
        for feat_key in self.preextracted_rgbd_feat_keys.values():
            obs[feat_key] = samples[feat_key].to(device, non_blocking=True)
        for spatial_key in self.preextracted_rgbd_spatial_keys.values():
            obs[spatial_key] = samples[spatial_key].to(device, non_blocking=True)

        return {
            "obs": obs,
            "action": action,
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
