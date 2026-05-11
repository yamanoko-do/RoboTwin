from typing import Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from mdm.model.v2 import MDMModel
from mdm.model.dinov2_rgbd.layers.block import attn_bias_cache as _attn_bias_cache
from mdm.model.modules_decoder import ResidualConvBlock
from mdm.utils.geo import normalized_view_plane_uv


class SpatialAggregationHead(nn.Module):
    """Aggregates spatial feature maps into a 1D feature vector.

    Follows the MDMModel preparation: features + CLS broadcast + UV concat,
    then compresses via Conv blocks + global average pooling + projection.
    """

    def __init__(
        self,
        in_channels: int = 1026,
        hidden_channels: int = 256,
        feature_dim: int = 512,
        num_blocks: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else hidden_channels
            blocks.append(ResidualConvBlock(
                in_channels=in_ch,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=3,
                padding_mode="replicate",
                activation=activation,
                in_norm="layer_norm",
                hidden_norm="group_norm",
            ))
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(hidden_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — spatial features with CLS injected + UV concatenated
        Returns:
            (B, feature_dim)
        """
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.projection(x)


class LingBotDepthEncoder(nn.Module):
    """Wraps MDMModel (LingBot-Depth) as a feature encoder.

    Supports two modes:
      - CLS-only (legacy): extracts CLS token and projects to feature_dim.
      - Spatial (default): follows MDMModel pipeline — features + CLS broadcast + UV
        concat → SpatialAggregationHead → feature_dim.

    The DINOv2 ViT backbone is frozen; all other parameters (projection head,
    SpatialAggregationHead) are trainable.
    """

    def __init__(
        self,
        pretrained_path: str = "/mnt/data/cpfs/yama/lingbotDepth/model.pt",
        feature_dim: int = 512,
        freeze_backbone: bool = True,
        resolution_level: int = 5,
        use_fp16: bool = True,
        use_spatial_features: bool = True,
        pool_h: int = 6,
        pool_w: int = 8,
        aspect_ratio: float = 4.0 / 3.0,
        head_hidden_channels: int = 256,
        head_num_blocks: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16
        self.use_spatial_features = use_spatial_features
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.aspect_ratio = aspect_ratio

        # Load pretrained model
        self.mdm_model = MDMModel.from_pretrained(pretrained_path)

        # Freeze backbone
        if self.freeze_backbone:
            self.mdm_model.requires_grad_(False)
            self.mdm_model.eval()

        cls_dim = self.mdm_model.encoder.dim_features  # 1024 for ViT-L

        if self.use_spatial_features:
            # in_channels = cls_dim (spatial features) + 2 (UV channels)
            self.spatial_head = SpatialAggregationHead(
                in_channels=cls_dim + 2,
                hidden_channels=head_hidden_channels,
                feature_dim=feature_dim,
                num_blocks=head_num_blocks,
            )
            # Cache UV grid for pre-extracted path (deterministic, same every call)
            self._uv_cache = {}
        else:
            # Legacy CLS-only projection
            self.projection = nn.Linear(cls_dim, feature_dim)

    def _get_num_tokens(self) -> int:
        min_tokens, max_tokens = self.mdm_model.num_tokens_range
        return int(min_tokens + (self.resolution_level / 9) * (max_tokens - min_tokens))

    def _get_uv(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Compute UV coordinate grid at the pooled resolution (deterministic)."""
        uv = normalized_view_plane_uv(
            width=self.pool_w,
            height=self.pool_h,
            aspect_ratio=self.aspect_ratio,
            dtype=dtype,
            device=device,
        )
        # (H, W, 2) -> (2, H, W) -> (1, 2, H, W) -> (B, 2, H, W)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return uv

    @torch.no_grad()
    def extract_cls_token(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run frozen backbone only and return raw CLS token (B, cls_dim).

        Used during pre-extraction to save backbone features to zarr.
        """
        num_tokens = self._get_num_tokens()
        _attn_bias_cache.clear()

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
        return cls_token.float()

    @torch.no_grad()
    def extract_spatial_features(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run frozen backbone and return pooled spatial features + CLS token.

        Pre-extraction only — no CLS injection, no UV, no projection.
        CLS injection happens at training time so the same cls_token serves
        both CLS-only and spatial paths.

        Returns:
            features_pooled: (B, cls_dim, pool_h, pool_w) — raw spatial features, pooled
            cls_token: (B, cls_dim) — global CLS token
        """
        num_tokens = self._get_num_tokens()
        _attn_bias_cache.clear()

        with torch.autocast(
            device_type=rgb.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_fp16 and rgb.dtype != torch.bfloat16,
        ):
            features, cls_token = self.mdm_model.forward_feat(
                image=rgb,
                num_tokens=num_tokens,
                depth=depth,
            )
        # features: (B, cls_dim, H, W) — pool to reduced resolution
        features_pooled = F.adaptive_avg_pool2d(features.float(), (self.pool_h, self.pool_w))
        cls_token = cls_token.float()
        return features_pooled, cls_token

    def _spatial_forward(
        self,
        features: torch.Tensor,
        cls_token: torch.Tensor,
    ) -> torch.Tensor:
        """Shared spatial processing for both online and pre-extracted paths.

        Args:
            features: (B, cls_dim, pool_h, pool_w) spatial features (pooled if online)
            cls_token: (B, cls_dim) CLS token
        Returns:
            (B, feature_dim)
        """
        # Inject CLS into spatial map (matching MDMModel v2.py line 121)
        features = features + cls_token[..., None, None]

        # Compute UV at pooled resolution
        uv = self._get_uv(features.shape[0], features.dtype, features.device)

        # Concatenate UV
        x = torch.cat([features, uv], dim=1)  # (B, cls_dim+2, pool_h, pool_w)

        # Run trainable spatial aggregation head
        return self.spatial_head(x)

    def forward(
        self,
        rgb: torch.Tensor = None,
        depth: Optional[torch.Tensor] = None,
        cls_token: Optional[torch.Tensor] = None,
        preextracted_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) float32 in [0, 1] range (NOT ImageNet normalized).
            depth: (B, 1, H, W) float32 in meters, or None for RGB-only mode.
            cls_token: (B, cls_dim) pre-extracted CLS token.
            preextracted_features: (B, cls_dim, pool_h, pool_w) pre-extracted spatial features.

        Modes:
            1. forward(rgb, depth) — online: backbone → spatial pipeline
            2. forward(cls_token=..., preextracted_features=...) — pre-extracted spatial
            3. forward(cls_token=...) — legacy CLS-only (when use_spatial_features=False
               or when spatial features not available)

        Returns:
            (B, feature_dim) feature vector.
        """
        if self.use_spatial_features:
            # --- Spatial mode ---
            if preextracted_features is not None and cls_token is not None:
                # Pre-extracted spatial path: inject CLS → UV → spatial_head
                return self._spatial_forward(preextracted_features, cls_token)

            if rgb is not None:
                # Online path: run backbone, pool, then spatial pipeline
                num_tokens = self._get_num_tokens()
                _attn_bias_cache.clear()

                ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
                with ctx:
                    with torch.autocast(
                        device_type=rgb.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_fp16 and rgb.dtype != torch.bfloat16,
                    ):
                        features, cls_token = self.mdm_model.forward_feat(
                            image=rgb,
                            num_tokens=num_tokens,
                            depth=depth,
                        )
                features = F.adaptive_avg_pool2d(features.float(), (self.pool_h, self.pool_w))
                cls_token = cls_token.float()
                return self._spatial_forward(features, cls_token)

            # Fallback: cls_token only (no spatial features available)
            # Use zero spatial features and inject CLS — the spatial head will
            # still see CLS broadcast over the map + UV coordinates.
            if cls_token is not None:
                zero_features = torch.zeros(
                    cls_token.shape[0], cls_token.shape[1],
                    self.pool_h, self.pool_w,
                    dtype=cls_token.dtype, device=cls_token.device,
                )
                return self._spatial_forward(zero_features, cls_token)

        else:
            # --- Legacy CLS-only mode ---
            if cls_token is not None:
                return self.projection(cls_token.float())

            num_tokens = self._get_num_tokens()
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
            cls_token = cls_token.float()
            return self.projection(cls_token)

        raise ValueError("Invalid argument combination for LingBotDepthEncoder.forward()")
