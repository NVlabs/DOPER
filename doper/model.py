# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DOPER unified pose network.

The SimCC/GAU/RTMCC keypoint head (``ScaleNorm``, ``GAU``, ``SimCCHead``) is a
minimal port of the corresponding MMPose (OpenMMLab) RTMPose components,
which are released under the Apache License 2.0 (Copyright 2018-2020 Open-MMLab);
see THIRD_PARTY_NOTICES.md.

A single shared backbone feeds two paths:

* a **full-image path** with an FPN neck down to stride 4, carrying a
  CenterNet-style detection head (heatmap / wh / offset) and a foreground mask
  head. Real-world bounding-box and mask labels can supervise this branch.
* a **top-down keypoint path** that receives image crops (default 512 x 512)
  and runs them through the *same* backbone weights. The default heatmap head
  uses a dedicated FPN to produce stride-4 2D logits for each keypoint.
  SimCC remains an optional head. Cropping happens outside this module;
  heatmap decoding in ``doper.targets`` uses DARK-style subpixel refinement
  by default.

The paths share backbone weights but compute separate feature maps and use
separate FPNs and prediction heads. An optional CAD-curve head also processes
the crop backbone features. An optional rotation auxiliary classifier pools
the keypoint FPN features for coarse orientation supervision.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional

from .config import ModelConfig
from .rotation_aux import CosineRotationHead, pooled_rotation_features, uniform_rotation_bins

# Feature strides every backbone must expose (C2..C5).
FEATURE_STRIDES = (4, 8, 16, 32)
# Upper bound on GroupNorm groups used in the conv towers.
MAX_GROUPNORM_GROUPS = 32
# Focal-loss friendly bias init: -log((1 - 0.1) / 0.1), i.e. a 0.1 prior.
FOCAL_BIAS_INIT = -2.19
SIMCC_FINAL_KERNEL_SIZE = 7
SIMCC_FINAL_PADDING = 3
# Total stride of the deepest backbone feature used by the SimCC head.
SIMCC_FEATURE_STRIDE = 32
DEFAULT_GAU_HIDDEN_DIMS = 256
DEFAULT_GAU_S = 128
DEFAULT_GAU_EXPANSION = 2
SCALE_NORM_EPS = 1e-5
CONV_TOWER_NUM_CONVS = 2
WH_AND_OFFSET_CHANNELS = 2

_RESNET_CHANNELS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
}


class ResNetBackbone(nn.Module):
    """Shared torchvision ResNet returning C2..C5 (strides 4/8/16/32)."""

    def __init__(self, name: str, pretrained: bool) -> None:
        """Build the torchvision ResNet ``name``.

        Args:
            name: One of ``resnet18``, ``resnet34`` or ``resnet50``.
            pretrained: Load the default torchvision weights when true.

        Raises:
            ValueError: If ``name`` is not a supported ResNet.
        """
        super().__init__()
        if name not in _RESNET_CHANNELS:
            raise ValueError(f"unsupported backbone {name!r}")
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = _RESNET_CHANNELS[name]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the C2..C5 feature maps for an image batch ``x``."""
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class TimmFeatureBackbone(nn.Module):
    """timm backbone returning four feature maps at strides 4/8/16/32."""

    def __init__(self, name: str, pretrained: bool) -> None:
        """Create the timm feature extractor ``name``.

        Args:
            name: timm model identifier.
            pretrained: Load pretrained weights when true.

        Raises:
            ImportError: If timm is not installed.
            RuntimeError: If timm cannot create the model.
            ValueError: If the feature strides are not ``FEATURE_STRIDES``.
        """
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"backbone {name!r} requires timm; install project requirements"
            ) from exc
        try:
            self.model: Any = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as exc:
            init = "pretrained" if pretrained else "uninitialized"
            raise RuntimeError(f"failed to create {init} timm backbone {name!r}: {exc}") from exc
        reductions = tuple(int(x) for x in self.model.feature_info.reduction())
        if reductions != FEATURE_STRIDES:
            raise ValueError(
                f"backbone {name!r} has feature strides {reductions}; DOPER requires (4, 8, 16, 32)"
            )
        c2, c3, c4, c5 = (int(x) for x in self.model.feature_info.channels())
        self.out_channels = (c2, c3, c4, c5)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the four feature maps for an image batch ``x``."""
        return list(self.model(x))


def build_backbone(name: str, pretrained: bool) -> ResNetBackbone | TimmFeatureBackbone:
    """Construct a supported shared backbone without changing ResNet key names.

    Args:
        name: ResNet or timm ConvNeXt identifier.
        pretrained: Load pretrained weights where the backbone supports it.

    Returns:
        The backbone module; it exposes ``out_channels`` for C2..C5.

    Raises:
        ValueError: If ``name`` is not a supported backbone.
    """
    if name in _RESNET_CHANNELS:
        return ResNetBackbone(name, pretrained)
    if name.startswith("convnext"):
        return TimmFeatureBackbone(name, pretrained)
    raise ValueError(
        f"unsupported backbone {name!r}; expected a torchvision ResNet or timm ConvNeXt identifier"
    )


def _focal_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Return a 1x1 classification conv with the focal-loss prior bias."""
    conv = nn.Conv2d(in_channels, out_channels, 1)
    assert conv.bias is not None
    nn.init.constant_(conv.bias, FOCAL_BIAS_INIT)
    return conv


class FPNToStride4(nn.Module):
    """Top-down FPN that fuses C2..C5 into a single stride-4 feature map."""

    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int) -> None:
        """Create the lateral 1x1 convolutions and the 3x3 smoothing convolution."""
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Fuse the C2..C5 maps ``feats`` into one stride-4 map."""
        laterals = [conv(f) for conv, f in zip(self.lateral, feats, strict=False)]
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            x = laterals[i] + functional.interpolate(x, size=laterals[i].shape[-2:], mode="nearest")
        return self.smooth(x)


def _conv_tower(in_ch: int, out_ch: int, num_convs: int = CONV_TOWER_NUM_CONVS) -> nn.Sequential:
    """Stack ``num_convs`` 3x3 Conv-GroupNorm-ReLU blocks."""
    layers: list[nn.Module] = []
    ch = in_ch
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(MAX_GROUPNORM_GROUPS, out_ch), out_ch),
            nn.ReLU(inplace=True),
        ]
        ch = out_ch
    return nn.Sequential(*layers)


class ScaleNorm(nn.Module):
    """RMS-style norm over the last dim with a single learnable gain.

    Matches MMPose's ``ScaleNorm`` (used by the RTMCC head/GAU): normalizes each
    token to unit RMS, scaled by ``dim ** -0.5`` and a scalar gain ``g``.
    """

    def __init__(self, dim: int, eps: float = SCALE_NORM_EPS) -> None:
        """Create the norm for feature dimension ``dim`` with stabilizer ``eps``."""
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension."""
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class GAU(nn.Module):
    """Gated Attention Unit with self-attention across the keypoint tokens.

    Faithful minimal port of MMPose's ``RTMCCBlock`` (self-attn mode): a single
    projection produces the gate ``u``, value ``v``, and a shared ``base`` that
    is split into query/key via learnable per-dim affine (gamma/beta). The
    attention kernel is a squared-ReLU (not softmax), and the token axis is the
    ``K`` keypoints, so every keypoint's prediction is conditioned on all the
    others - the rigid-object geometric prior the old independent-linear head
    lacked. Residual connection keeps it drop-in.
    """

    def __init__(
        self, dim: int, s: int = DEFAULT_GAU_S, expansion: int = DEFAULT_GAU_EXPANSION
    ) -> None:
        """Create the unit.

        Args:
            dim: Token embedding dimension.
            s: Shared query/key base dimension.
            expansion: Expansion factor of the gate and value projections.
        """
        super().__init__()
        e = dim * expansion
        self.ln = ScaleNorm(dim)
        self.uv = nn.Linear(dim, 2 * e + s, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, s))
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.o = nn.Linear(e, dim, bias=False)
        self.e = e
        self.s = s
        self.sqrt_s = float(s) ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated attention across keypoint tokens of shape ``(B, K, dim)``."""
        shortcut = x
        x = self.ln(x)
        uv = functional.silu(self.uv(x))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :, :] + self.beta
        q, k = torch.unbind(base, dim=2)  # each (B, K, s)
        qk = torch.bmm(q, k.transpose(1, 2))  # (B, K, K) across keypoints
        kernel = torch.square(functional.relu(qk / self.sqrt_s))
        x = u * torch.bmm(kernel, v)
        return shortcut + self.o(x)


class SimCCHead(nn.Module):
    """RTMCC-style SimCC keypoint head.

    Backbone feature -> large-kernel conv (one channel per keypoint) -> flatten
    spatial -> ScaleNorm + embedding Linear -> GAU (cross-keypoint self-attn) ->
    separate x / y bin classifiers. This replaces the previous bare
    ``Linear(h*w -> bins)`` per keypoint, which had no shared embedding, no
    nonlinearity, and no cross-keypoint reasoning.
    """

    def __init__(
        self,
        in_ch: int,
        num_keypoints: int,
        feat_hw: int,
        bins: int,
        hidden_dims: int = DEFAULT_GAU_HIDDEN_DIMS,
        gau_s: int = DEFAULT_GAU_S,
        gau_expansion: int = DEFAULT_GAU_EXPANSION,
    ) -> None:
        """Create the head.

        Args:
            in_ch: Channels of the deepest backbone feature.
            num_keypoints: Number of keypoints ``K``.
            feat_hw: Side length of the (square) deepest feature map.
            bins: Number of SimCC bins per axis.
            hidden_dims: Keypoint embedding dimension.
            gau_s: GAU shared query/key dimension.
            gau_expansion: GAU expansion factor.
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.final = nn.Conv2d(
            in_ch,
            num_keypoints,
            kernel_size=SIMCC_FINAL_KERNEL_SIZE,
            padding=SIMCC_FINAL_PADDING,
        )
        flat = feat_hw * feat_hw
        self.mlp = nn.Sequential(
            ScaleNorm(flat),
            nn.Linear(flat, hidden_dims, bias=False),
        )
        self.gau = GAU(hidden_dims, s=gau_s, expansion=gau_expansion)
        self.cls_x = nn.Linear(hidden_dims, bins, bias=False)
        self.cls_y = nn.Linear(hidden_dims, bins, bias=False)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(simcc_x, simcc_y)`` logits, each ``(B, K, bins)``."""
        x = self.final(feat)  # (B, K, h, w)
        x = torch.flatten(x, 2)  # (B, K, h*w)
        x = self.mlp(x)  # (B, K, hidden)
        x = self.gau(x)  # (B, K, hidden)
        return self.cls_x(x), self.cls_y(x)


class KeypointHeatmapHead(nn.Module):
    """Joint 2D keypoint distributions on a stride-4 crop feature map.

    A dedicated top-down FPN preserves high-resolution C2 detail while fusing
    the semantic context from C3..C5. Each output channel is one keypoint's
    spatial categorical distribution. Unlike independent SimCC x/y marginals,
    every selectable peak is therefore a valid joint 2D location.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        neck_channels: int,
        head_channels: int,
        num_keypoints: int,
    ) -> None:
        """Create the FPN neck, conv tower and 1x1 output layer."""
        super().__init__()
        self.neck = FPNToStride4(in_channels, neck_channels)
        self.tower = _conv_tower(neck_channels, head_channels, num_convs=CONV_TOWER_NUM_CONVS)
        self.out = nn.Conv2d(head_channels, num_keypoints, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Return ``(B, K, H, W)`` keypoint heatmap logits from C2..C5 ``feats``."""
        return self.out(self.tower(self.neck(feats)))


class NamedCurveHead(nn.Module):
    """Independent stride-4 heatmaps for a fixed ordered CAD-curve vocabulary."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        neck_channels: int,
        head_channels: int,
        num_curves: int,
    ) -> None:
        """Create the head.

        Raises:
            ValueError: If ``num_curves`` is not positive.
        """
        super().__init__()
        if num_curves <= 0:
            raise ValueError("num_curves must be positive when curve head is enabled")
        self.neck = FPNToStride4(in_channels, neck_channels)
        self.tower = _conv_tower(neck_channels, head_channels, num_convs=CONV_TOWER_NUM_CONVS)
        self.out = _focal_conv(head_channels, num_curves)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Return ``(B, C, H, W)`` curve heatmap logits from C2..C5 ``feats``."""
        return self.out(self.tower(self.neck(feats)))


class DoperNet(nn.Module):
    """Shared-backbone network with optional curve and rotation auxiliary heads."""

    # Set by ``doper.infer.load_model`` from the checkpoint metadata.
    keypoints_3d_fingerprint: str | None = None
    rotation_bins: torch.Tensor

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the network described by ``cfg``.

        Raises:
            ValueError: If ``cfg.keypoint_head`` is neither ``simcc`` nor ``heatmap``.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.backbone, cfg.pretrained)
        chans = self.backbone.out_channels
        self.neck = FPNToStride4(chans, cfg.neck_channels)

        nc = cfg.neck_channels
        hc = cfg.head_channels
        self.det_tower = _conv_tower(nc, hc, num_convs=CONV_TOWER_NUM_CONVS)
        self.hm_head = _focal_conv(hc, cfg.num_classes)
        self.wh_head = nn.Conv2d(hc, WH_AND_OFFSET_CHANNELS, 1)
        self.offset_head = nn.Conv2d(hc, WH_AND_OFFSET_CHANNELS, 1)

        self.mask_tower = _conv_tower(nc, hc, num_convs=CONV_TOWER_NUM_CONVS)
        self.mask_head = nn.Conv2d(hc, 1, 1)

        if cfg.keypoint_head == "simcc":
            feat_hw = cfg.crop_size // SIMCC_FEATURE_STRIDE
            bins = int(round(cfg.crop_size * cfg.simcc_split_ratio))
            self.kpt_head = SimCCHead(
                chans[-1],
                cfg.num_keypoints,
                feat_hw,
                bins,
                hidden_dims=cfg.gau_hidden_dims,
                gau_s=cfg.gau_s,
                gau_expansion=cfg.gau_expansion,
            )
        elif cfg.keypoint_head == "heatmap":
            self.kpt_head = KeypointHeatmapHead(
                chans,
                cfg.neck_channels,
                cfg.head_channels,
                cfg.num_keypoints,
            )
        else:
            raise ValueError(
                f"unsupported keypoint_head {cfg.keypoint_head!r}; expected 'simcc' or 'heatmap'"
            )

        if cfg.rotation_aux_enabled:
            if cfg.keypoint_head != "heatmap":
                raise ValueError("rotation auxiliary requires a heatmap keypoint head")
            self.rotation_head = CosineRotationHead(
                cfg.neck_channels * cfg.rotation_pool_size**2,
                cfg.rotation_num_bins,
                cfg.rotation_logit_scale,
                cfg.rotation_pool_size,
            )
            self.register_buffer(
                "rotation_bins", uniform_rotation_bins(cfg.rotation_num_bins, cfg.rotation_bin_seed)
            )

        if cfg.curve_head_enabled:
            self.curve_head = NamedCurveHead(
                chans,
                cfg.neck_channels,
                cfg.curve_head_channels,
                cfg.num_curves,
            )

    # -- full-image detection + mask path -----------------------------------
    def forward_det(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full-image path and return ``heatmap``, ``wh``, ``offset`` and ``mask``."""
        feats = self.backbone(image)
        p2 = self.neck(feats)
        det = self.det_tower(p2)
        mask_feat = self.mask_tower(p2)
        return {
            "heatmap": self.hm_head(det),
            "wh": self.wh_head(det),
            "offset": self.offset_head(det),
            "mask": self.mask_head(mask_feat),
        }

    # -- top-down keypoint path (shared backbone) ---------------------------
    def forward_kpt(self, crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run only the keypoint head on ``crop``.

        Returns:
            ``(simcc_x, simcc_y)`` for the SimCC head, else the heatmap logits.
        """
        feats = self.backbone(crop)
        return self._forward_kpt_features(feats)

    def _forward_kpt_features(
        self, feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Dispatch backbone features to the configured keypoint head."""
        if self.cfg.keypoint_head == "simcc":
            return self.kpt_head(feats[-1])
        return self.kpt_head(feats)

    def forward_crop(self, crop: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run every enabled crop head from one shared-backbone feature pass."""
        feats = self.backbone(crop)
        if self.cfg.rotation_aux_enabled:
            assert isinstance(self.kpt_head, KeypointHeatmapHead)
            fused = self.kpt_head.neck(feats)
            keypoint_out = self.kpt_head.out(self.kpt_head.tower(fused))
        else:
            keypoint_out = self._forward_kpt_features(feats)
        out: dict[str, torch.Tensor] = {}
        if isinstance(keypoint_out, tuple):
            out["simcc_x"], out["simcc_y"] = keypoint_out
        else:
            out["keypoint_heatmap"] = keypoint_out
        if self.cfg.rotation_aux_enabled:
            out["rotation_logits"] = self.rotation_head(
                pooled_rotation_features(fused, self.cfg.rotation_pool_size)
            )
            out["rotation_bins"] = self.rotation_bins
        if self.cfg.curve_head_enabled:
            out["curve_heatmap"] = self.curve_head(feats)
        return out

    def forward(self, det_image: torch.Tensor, crop_image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the detection path on ``det_image`` and every crop head on ``crop_image``."""
        out = self.forward_det(det_image)
        out.update(self.forward_crop(crop_image))
        return out
