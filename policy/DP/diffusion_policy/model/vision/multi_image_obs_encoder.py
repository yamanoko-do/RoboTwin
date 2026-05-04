from typing import Dict, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


def _shape_match(actual, expected):
    """Check shape compatibility where -1 in expected matches any dimension."""
    if len(actual) != len(expected):
        return False
    return all(e == -1 or a == e for a, e in zip(actual, expected))


class DenormalizeRGB(nn.Module):
    """Undo LinearNormalizer's [0,1] -> [-1,1] to restore [0,1] range.

    The pipeline normalizes RGB via LinearNormalizer (scale=2, offset=-1)
    before passing to MultiImageObsEncoder.  LingBot-Depth expects [0,1]
    input because it applies ImageNet normalization internally.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) / 2.0


class DepthScaleTransform(nn.Module):
    """Convert depth units (e.g. millimeters -> meters)."""

    def __init__(self, scale: float = 1000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


class MultiImageObsEncoder(ModuleAttrMixin):

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        depth_model: Union[nn.Module, Dict[str, nn.Module], None] = None,
        rgbd_model: Union[nn.Module, None] = None,
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # use single depth model for all depth inputs
        share_depth_model: bool = False,
        # use single rgbd model for all rgbd inputs
        share_rgbd_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        # rgbd key pairs: maps rgb_key -> depth_key (None for RGB-only)
        rgbd_key_pairs: Optional[Dict[str, Optional[str]]] = None,
        # separate spatial transforms for rgbd pathway
        rgbd_resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        rgbd_crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        # depth scale (divide by this to get meters)
        depth_scale: float = 1000.0,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes depth input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        depth_keys = list()
        low_dim_keys = list()
        rgbd_keys = list()
        key_model_map = nn.ModuleDict()
        key_depth_model_map = nn.ModuleDict()
        key_rgbd_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_depth_transform_map = nn.ModuleDict()
        key_rgbd_transform_map = nn.ModuleDict()
        key_depth_rgbd_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        if share_depth_model and depth_model is not None:
            assert isinstance(depth_model, nn.Module)
            key_depth_model_map["depth"] = depth_model

        if share_rgbd_model and rgbd_model is not None:
            assert isinstance(rgbd_model, nn.Module)
            key_rgbd_model_map["rgbd"] = rgbd_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == "depth":
                depth_keys.append(key)
                # configure model for this key
                this_depth_model = None
                if depth_model is not None and not share_depth_model:
                    if isinstance(depth_model, dict):
                        this_depth_model = depth_model[key]
                    else:
                        assert isinstance(depth_model, nn.Module)
                        this_depth_model = copy.deepcopy(depth_model)

                if this_depth_model is not None:
                    if use_group_norm:
                        this_depth_model = replace_submodules(
                            root_module=this_depth_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_depth_model_map[key] = this_depth_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))

                # no ImageNet normalization for depth
                this_transform = nn.Sequential(this_resizer, this_randomizer)
                key_depth_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        depth_keys = sorted(depth_keys)
        low_dim_keys = sorted(low_dim_keys)

        # === Build rgbd pathway ===
        # rgbd_key_pairs maps rgb_key -> depth_key (None = RGB-only for LingBotDepth)
        if rgbd_key_pairs is not None:
            for rgb_key, depth_key in rgbd_key_pairs.items():
                rgbd_keys.append(rgb_key)

                # configure rgbd model for this key
                if not share_rgbd_model:
                    this_rgbd_model = copy.deepcopy(rgbd_model)
                    key_rgbd_model_map[rgb_key] = this_rgbd_model

                # build rgbd RGB transform: resize -> crop -> DenormalizeRGB
                # (NO ImageNet norm — LingBotDepth handles it internally)
                rgb_shape = key_shape_map[rgb_key]
                input_shape = rgb_shape
                this_resizer = nn.Identity()
                if rgbd_resize_shape is not None:
                    if isinstance(rgbd_resize_shape, dict):
                        h, w = rgbd_resize_shape[rgb_key]
                    else:
                        h, w = rgbd_resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (rgb_shape[0], h, w)

                this_randomizer = nn.Identity()
                if rgbd_crop_shape is not None:
                    if isinstance(rgbd_crop_shape, dict):
                        h, w = rgbd_crop_shape[rgb_key]
                    else:
                        h, w = rgbd_crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                this_rgbd_transform = nn.Sequential(
                    this_resizer,
                    this_randomizer,
                    DenormalizeRGB(),  # undo [-1,1] -> [0,1] for LingBotDepth
                )
                key_rgbd_transform_map[rgb_key] = this_rgbd_transform

                # build depth transform for rgbd pathway (if depth_key provided)
                if depth_key is not None:
                    depth_shape = key_shape_map[depth_key]
                    input_shape = depth_shape
                    this_depth_resizer = nn.Identity()
                    if rgbd_resize_shape is not None:
                        if isinstance(rgbd_resize_shape, dict):
                            h, w = rgbd_resize_shape.get(depth_key, rgbd_resize_shape.get(rgb_key, rgbd_resize_shape))
                        else:
                            h, w = rgbd_resize_shape
                        this_depth_resizer = torchvision.transforms.Resize(size=(h, w))
                        input_shape = (depth_shape[0], h, w)

                    this_depth_randomizer = nn.Identity()
                    if rgbd_crop_shape is not None:
                        if isinstance(rgbd_crop_shape, dict):
                            h, w = rgbd_crop_shape.get(depth_key, rgbd_crop_shape.get(rgb_key, rgbd_crop_shape))
                        else:
                            h, w = rgbd_crop_shape
                        if random_crop:
                            this_depth_randomizer = CropRandomizer(
                                input_shape=input_shape,
                                crop_height=h,
                                crop_width=w,
                                num_crops=1,
                                pos_enc=False,
                            )
                        else:
                            this_depth_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                    this_depth_rgbd_transform = nn.Sequential(
                        this_depth_resizer,
                        this_depth_randomizer,
                        DepthScaleTransform(depth_scale),
                    )
                    key_depth_rgbd_transform_map[depth_key] = this_depth_rgbd_transform

        rgbd_keys = sorted(rgbd_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_depth_model_map = key_depth_model_map
        self.key_rgbd_model_map = key_rgbd_model_map
        self.key_transform_map = key_transform_map
        self.key_depth_transform_map = key_depth_transform_map
        self.key_rgbd_transform_map = key_rgbd_transform_map
        self.key_depth_rgbd_transform_map = key_depth_rgbd_transform_map
        self.share_rgb_model = share_rgb_model
        self.share_depth_model = share_depth_model
        self.share_rgbd_model = share_rgbd_model
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.low_dim_keys = low_dim_keys
        self.rgbd_keys = rgbd_keys
        self.rgbd_key_pairs = rgbd_key_pairs or {}
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert _shape_match(img.shape[1:], self.key_shape_map[key])
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert _shape_match(img.shape[1:], self.key_shape_map[key])
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process depth input
        if self.share_depth_model and len(self.depth_keys) > 0:
            # pass all depth obs to shared depth model
            if "depth" in self.key_depth_model_map:
                imgs = list()
                for key in self.depth_keys:
                    img = obs_dict[key]
                    if batch_size is None:
                        batch_size = img.shape[0]
                    else:
                        assert batch_size == img.shape[0]
                    assert _shape_match(img.shape[1:], self.key_shape_map[key])
                    img = self.key_depth_transform_map[key](img)
                    imgs.append(img)
                # (N*B,C,H,W)
                imgs = torch.cat(imgs, dim=0)
                # (N*B,D)
                feature = self.key_depth_model_map["depth"](imgs)
                # (N,B,D)
                feature = feature.reshape(-1, batch_size, *feature.shape[1:])
                # (B,N,D)
                feature = torch.moveaxis(feature, 0, 1)
                # (B,N*D)
                feature = feature.reshape(batch_size, -1)
                features.append(feature)
        else:
            # run each depth obs to independent models
            for key in self.depth_keys:
                if key not in self.key_depth_model_map:
                    # depth_model not provided, skip depth features
                    continue
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert _shape_match(img.shape[1:], self.key_shape_map[key])
                img = self.key_depth_transform_map[key](img)
                feature = self.key_depth_model_map[key](img)
                features.append(feature)

        # process rgbd input (joint RGB+Depth encoding via LingBotDepth)
        if len(self.rgbd_keys) > 0:
            if self.share_rgbd_model:
                # batch all rgbd views through shared model
                rgbd_imgs = list()
                rgbd_depths = list()
                has_depth = False
                for key in self.rgbd_keys:
                    rgb = obs_dict[key]
                    if batch_size is None:
                        batch_size = rgb.shape[0]
                    else:
                        assert batch_size == rgb.shape[0]
                    rgb = self.key_rgbd_transform_map[key](rgb)
                    rgbd_imgs.append(rgb)

                    depth_key = self.rgbd_key_pairs.get(key)
                    if depth_key is not None and depth_key in obs_dict:
                        depth = obs_dict[depth_key]
                        depth = self.key_depth_rgbd_transform_map[depth_key](depth)
                        rgbd_depths.append(depth)
                        has_depth = True
                    else:
                        rgbd_depths.append(None)

                # Concat RGB: (N*B, 3, H, W)
                rgbd_imgs_cat = torch.cat(rgbd_imgs, dim=0)
                # Concat depth: (N*B, 1, H, W) or None
                if has_depth:
                    # For views without depth, use zeros
                    first_shape = rgbd_imgs_cat.shape
                    rgbd_depths_cat = torch.cat([
                        d if d is not None
                        else torch.zeros(first_shape[0] // len(rgbd_imgs), 1, first_shape[2], first_shape[3],
                                         device=first_shape.device if hasattr(first_shape, 'device') else rgbd_imgs_cat.device,
                                         dtype=rgbd_imgs_cat.dtype)
                        for d in rgbd_depths
                    ], dim=0)
                else:
                    rgbd_depths_cat = None

                # Forward through shared rgbd model
                feature = self.key_rgbd_model_map["rgbd"](rgbd_imgs_cat, depth=rgbd_depths_cat)
                # (N*B, D) -> (N, B, D) -> (B, N, D) -> (B, N*D)
                feature = feature.reshape(-1, batch_size, *feature.shape[1:])
                feature = torch.moveaxis(feature, 0, 1)
                feature = feature.reshape(batch_size, -1)
                features.append(feature)
            else:
                # per-key rgbd processing
                for key in self.rgbd_keys:
                    rgb = obs_dict[key]
                    if batch_size is None:
                        batch_size = rgb.shape[0]
                    else:
                        assert batch_size == rgb.shape[0]
                    rgb = self.key_rgbd_transform_map[key](rgb)

                    depth_key = self.rgbd_key_pairs.get(key)
                    depth = None
                    if depth_key is not None and depth_key in obs_dict:
                        depth = obs_dict[depth_key]
                        depth = self.key_depth_rgbd_transform_map[depth_key](depth)

                    feature = self.key_rgbd_model_map[key](rgb, depth=depth)
                    features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert _shape_match(data.shape[1:], self.key_shape_map[key])
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            # replace -1 placeholders with 1 for dry-run shape inference
            shape = tuple(1 if s == -1 else s for s in shape)
            this_obs = torch.zeros((batch_size, ) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
