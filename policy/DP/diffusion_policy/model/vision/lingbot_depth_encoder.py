from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from mdm.model.v2 import MDMModel
from mdm.model.dinov2_rgbd.layers.block import attn_bias_cache as _attn_bias_cache


class LingBotDepthEncoder(nn.Module):
    """Wraps MDMModel (LingBot-Depth) as a feature encoder.

    Extracts the CLS token from the DINOv2 ViT backbone and projects it
    to a configurable feature dimension. Supports both RGB+Depth and
    RGB-only modes (pass depth=None for monocular estimation).

    The backbone is frozen by default — only the projection head is trainable.
    """

    def __init__(
        self,
        pretrained_path: str = "/mnt/data/cpfs/yama/lingbotDepth/model.pt",
        feature_dim: int = 512,
        freeze_backbone: bool = True,
        resolution_level: int = 5,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16

        # Load pretrained model
        self.mdm_model = MDMModel.from_pretrained(pretrained_path)

        # Freeze backbone
        if self.freeze_backbone:
            self.mdm_model.requires_grad_(False)
            self.mdm_model.eval()

        # Projection head: CLS token (1024-dim for ViT-L) -> feature_dim
        cls_dim = self.mdm_model.encoder.dim_features
        self.projection = nn.Linear(cls_dim, feature_dim)

    def _get_num_tokens(self) -> int:
        min_tokens, max_tokens = self.mdm_model.num_tokens_range
        return int(min_tokens + (self.resolution_level / 9) * (max_tokens - min_tokens))

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) float32 in [0, 1] range (NOT ImageNet normalized).
            depth: (B, 1, H, W) float32 in meters, or None for RGB-only mode.

        Returns:
            (B, feature_dim) feature vector.
        """
        num_tokens = self._get_num_tokens()

        # Clear stale attn_bias_cache entries that may have been created
        # on a different device (e.g. CPU during output_shape() dry-run).
        _attn_bias_cache.clear()

        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            with torch.autocast(
                device_type=rgb.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_fp16 and rgb.dtype != torch.bfloat16,
            ):
                _, cls_token = self.mdm_model.forward_feat(
                    image=rgb,
                    num_tokens=num_tokens,
                    depth=depth,
                )
        # cls_token: (B, 1024) — cast back to float32 for projection
        cls_token = cls_token.float()
        return self.projection(cls_token)
